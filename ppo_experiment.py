# main.py
"""
This is the main script for training RL agents with CSE and DR using Ray RLlib.
"""

import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.tune.registry import register_env
from ray.tune.result import TRAINING_ITERATION

from environments.common import env_class_dict, env_lambda, get_env_config
from utils.cse import SampleEnhanceConnector
from utils.dr import GlobalDomainRandomizationCallback, LocalDomainRandomizationCallback
from utils.evaluation import custom_eval_function_single_env_runner
from utils.make_args import make_args
from utils.train_kwargs import get_model_config

torch, _ = try_import_torch()

if __name__ == "__main__":

    # for custom progress reporting
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    os.environ["RAY_AIR_RICH_LAYOUT"] = "1"

    # * Input kwargs
    input_args = make_args()

    ray.init()
    env_class = env_class_dict.get(input_args.env, None)
    algo_config = PPOConfig

    # * Environment Setup
    if input_args.env in env_class_dict:
        env_creator = env_lambda(input_args.env)
        # register env with rllib
        register_env(input_args.env, env_creator)
    else:
        raise ValueError(f"Environment {input_args.env} not recognized.")

    # Create environment config
    env_config = get_env_config(input_args)

    #! * Base Algorithm Configuration
    base_config = (
        algo_config()
        .environment(
            env=input_args.env,
            env_config=env_config,
        )
        .env_runners(
            rollout_fragment_length="auto",
            num_env_runners=1 if input_args.sample_parallel else 0,
            batch_mode="truncate_episodes",
            validate_env_runners_after_construction=False,
            num_envs_per_env_runner=(
                input_args.vectorization if input_args.vectorization > 1 else 1
            ),
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .rl_module(model_config=get_model_config(input_args))
    )

    #! Domain Randomization
    if input_args.dr == 1:
        base_config = base_config.callbacks(
            lambda: LocalDomainRandomizationCallback(
                env_config["context_values"],
                input_args.se_delta,
                default_eval=not input_args.custom_eval,
            )
        )
    elif input_args.dr == 2:
        base_config = base_config.callbacks(
            lambda: GlobalDomainRandomizationCallback(
                env_config["context_values"],
                env_class.context_low,
                env_class.context_high,
                default_eval=not input_args.custom_eval,
            )
        )

    #! Train kwargs
    train_kwargs = dict(lr=0.001, clip_param=0.2)

    #! Sample Enhancement
    if input_args.cse == 1:
        train_kwargs["learner_connector"] = (
            lambda obs_sp, act_sp: SampleEnhanceConnector(
                input_obs_space=obs_sp,
                input_act_space=act_sp,
                delta=dict(s=0, a=0, c=input_args.se_delta),
                action_mtd=input_args.se_action,
            )
        )
    elif input_args.cse == 2:
        train_kwargs["learner_connector"] = (
            lambda obs_sp, act_sp: SampleEnhanceConnector(
                input_obs_space=obs_sp,
                input_act_space=act_sp,
                delta=dict(s=0, a=0, c=input_args.se_delta),
                transition_fn=env_class.calculate_transition,
                reward_fn=env_class.calculate_reward,
                action_mtd=input_args.se_action,
            )
        )

    #! Custom evaluation
    base_config = base_config.training(**train_kwargs)

    if input_args.custom_eval:
        d = dict(
            custom_evaluation_function=custom_eval_function_single_env_runner,
            evaluation_config={"explore": False, "eval_iters": input_args.eval_iters},
        )
        if input_args.eval_parallel:
            d["evaluation_parallel_to_training"] = True
            d["evaluation_num_env_runners"] = 1
        if input_args.eval_interval > 0:
            d["evaluation_interval"] = input_args.eval_interval
    else:
        d = dict(
            evaluation_config={"explore": False},
            evaluation_duration="auto" if input_args.eval_parallel else 1,
        )
        if input_args.eval_parallel:
            d["evaluation_parallel_to_training"] = True
            d["evaluation_num_env_runners"] = 1
        if input_args.eval_interval > 0:
            d["evaluation_interval"] = input_args.eval_interval
    base_config = base_config.evaluation(**d)
    del d

    #! * Progress Reporter
    if input_args.custom_eval:
        neval_envs = env_config.get("perturbations")
        if neval_envs is not None:
            neval_envs = neval_envs.shape[0]
        else:
            neval_envs = 1
    else:
        neval_envs = 1
    if input_args.custom_eval:
        my_progress_reporter = tune.CLIReporter(
            metric_columns={
                TRAINING_ITERATION: "iter",
                "time_total_s": "total time (s)",
                NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                "env_runners/episode_return_mean": "train_mean_ret",
                **{
                    f"evaluation/env_runners/{i}/episode_return_mean": f"eval {i}"
                    for i in range(10)
                },
                "evaluation/env_runners/episode_return_mean": "eval_mean_ret",
            },
            max_report_frequency=30,  # seconds
        )
    else:
        my_progress_reporter = tune.CLIReporter(
            metric_columns={
                TRAINING_ITERATION: "iter",
                "time_total_s": "total time (s)",
                NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                "env_runners/episode_return_mean": "TrMeanRet",
                "env_runners/episode_len_mean": "TrMeanEpLen",
                "evaluation/env_runners/episode_return_mean": "EvMeanRet",
                "evaluation/env_runners/episode_len_mean": "EvMeanEpLen",
            },
            max_report_frequency=30,  # seconds
        )

    #! * Additional configs for training
    tune_config = tune.TuneConfig(
        num_samples=input_args.samples,
        max_concurrent_trials=input_args.max_concurrent,
    )

    checkpoint_dict = dict(checkpoint_at_end=True)
    if input_args.checkpoint_at_eval and input_args.eval_interval > 0:
        checkpoint_dict["checkpoint_frequency"] = input_args.eval_interval

    exp_name = input_args.env
    if input_args.cse:
        exp_name += f"_cse{input_args.cse}"
    elif input_args.dr:
        exp_name += f"_dr{input_args.dr}"
    else:
        exp_name += "_baseline"
    run_config = tune.RunConfig(
        name=exp_name,
        stop={"training_iteration": input_args.epochs},
        progress_reporter=my_progress_reporter,
        verbose=1,
        checkpoint_config=tune.CheckpointConfig(**checkpoint_dict),
    )
    tuner = tune.Tuner(
        base_config.algo_class,
        param_space=base_config,
        tune_config=tune_config,
        run_config=run_config,
    )

    #! * Log experiment parameters
    tmp = tuner._local_tuner.get_run_config()
    path = os.path.join(tmp.storage_path, tmp.name)

    # Save the input_kwargs in the experiment directory
    args_dict = vars(input_args)
    with open(os.path.join(path, "input_args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    # Log the run commands
    ymd, hms = datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(" ")
    runlog_path = os.path.join(os.getcwd(), "run logs", ymd + ".txt")
    mode = "a" if os.path.exists(runlog_path) else "x"
    suggested_name = [input_args.env]
    if input_args.cse:
        suggested_name.append(f"cse{input_args.cse}")
    if input_args.dr:
        suggested_name.append(f"dr{input_args.dr}")
    with open(runlog_path, mode) as f:
        f.write(" ".join(sys.argv) + "\n")
        f.write("\tPath: " + path + "\n")
        f.write("\tName: " + "_".join(suggested_name) + "\n")
        f.write("\tTime: " + hms + "\n")

    results_grid = tuner.fit()

    with open(runlog_path, "a") as f:
        f.write(
            "\tInfo: "
            + str(results_grid.num_errors)
            + " errors / "
            + str(len(results_grid))
            + " trials\n"
        )

    # * Plotting
    exp_identifier = results_grid[0].path.split("_")[-5]
    data = []
    for res in results_grid:
        df = res.metrics_dataframe
        data.append(df["env_runners/episode_return_mean"])

    df = pd.DataFrame(data).T
    mean = df.mean(axis=1)
    sem = df.sem(axis=1)
    plt.plot(df.index, mean)
    if not sem.hasnans:
        plt.fill_between(
            df.index,
            mean - sem,
            mean + sem,
            alpha=0.2,
        )
    plt.savefig(os.path.join(path, f"training_results_{exp_identifier}.png"))
    plt.clf()

    # * Evaluation plots
    if input_args.custom_eval:
        plot_data = [dict() for _ in range(neval_envs)]
        for tid, res in enumerate(results_grid):
            df = res.metrics_dataframe

            for idx in range(neval_envs):
                plot_data[idx][tid] = df[
                    f"evaluation/env_runners/{idx}/episode_return_mean"
                ]

        summary_df = pd.DataFrame(columns=["mean", "sem"])
        for idx, data in enumerate(plot_data):
            df = pd.DataFrame(data)

            mean = df.mean(axis=1)
            sem = df.sem(axis=1)
            plt.plot(mean, label=f"eval env {idx}")
            plt.fill_between(
                df.index,
                mean - sem,
                mean + sem,
                alpha=0.2,
            )
            summary_df.loc[idx] = {"mean": mean.iloc[-1], "sem": sem.iloc[-1]}

        print(summary_df)
        # Evaulation plot
        if neval_envs > 1:
            plt.legend()
        plt.savefig(os.path.join(path, f"evaluation_results_{exp_identifier}.png"))
        plt.clf()

        # Log results
        with open(runlog_path, "a") as f:
            f.write(
                "\tEval: "
                + ", ".join(
                    f"{summary_df.loc[idx]['mean']:.1f} +/- {summary_df.loc[idx]['sem']:.1f}"
                    for idx in range(neval_envs)
                )
                + "\n"
            )

        summary_df.to_csv(
            os.path.join(path, f"evaluation_summary_{exp_identifier}.csv"), index=False
        )

    ray.shutdown()

    time.sleep(1)
    print("Results saved to: ", path)
