# evaluate_policies.py
"""
This script evaluates policies for many contexts.
"""

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ray
import torch
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit
from ray.experimental import tqdm_ray
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_compile_config import TorchCompileConfig

from environments.common import env_class_dict, get_env_config, time_limit_dict

# * Variables

# Evaluation iteration parameters
NEPISODES = 64


def make_plot_args():
    parser = argparse.ArgumentParser(description="Evaluate policies across methods.")
    parser.add_argument(
        "--env",
        type=str,
        nargs="+",
        required=True,
        choices=list(env_class_dict.keys()) + ["bc_simple_dir"],
        help="Name of the environment to evaluate (e.g., 'simple0' or 'simple1').",
    )
    parser.add_argument(
        "--nmesh",
        type=int,
        default=5,
        help="Number of points to sample in the context space.",
    )
    parser.add_argument(
        "--save_fig",
        action="store_true",
        help="Save the figure instead of showing it.",
    )
    parser.add_argument(
        "--vectorization",
        type=int,
        default=32,
        help="Number of vectorized environments to use.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=["euler", "rk45", "solve_ivp"],
        help="ODE solver to use",
    )
    parser.add_argument(
        "--cse",
        type=int,
        help="Enable context sample enhancement. 0 for none. 1 for first ordre. 2 for second order",
        default=0,
    )

    parser.add_argument(
        "--base_context",
        type=float,
        nargs="+",
        help="Base context values for the environment.",
        default=[],
    )

    parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=-1,
        help="Checkpoint number to use. -1 for the latest checkpoint.",
    )
    parser.add_argument(
        "--bc",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Test mode. No saving.",
    )

    return parser.parse_args()


# Walk through the path and print the folders
def get_policy_paths(general_path: str, num: int) -> list[Path]:
    policy_paths = []
    for filename in os.listdir(general_path):
        if os.path.isdir(os.path.join(general_path, filename)):
            p1 = os.path.join(general_path, filename)
            tmp = os.listdir(p1)
            tmp = sorted([x for x in tmp if x.startswith("checkpoint_")])
            if len(tmp) == 0:
                print(f"No checkpoints found in {p1}")
                continue
            checkpoint_str = tmp[num]
            p = (
                Path(general_path)
                / filename
                / checkpoint_str
                / "learner_group"
                / "learner"
                / "rl_module"
                / "default_policy"
            )
            if os.path.exists(p):
                policy_paths.append(p)
    return sorted(policy_paths)


@torch.no_grad()
def get_returns(env, module, nepisodes: int):
    nenvs = env.num_envs
    if nepisodes % nenvs != 0:
        raise ValueError(
            f"Number of episodes {nepisodes} must be divisible by number of envs {nenvs}"
        )
    data = np.zeros(nepisodes)
    for ep_ind in range(nepisodes // nenvs):
        done = np.array([False] * nenvs)
        obs, info = env.reset()
        tot = 0
        while not np.all(done):
            # get action
            obs_dict = {Columns.OBS: torch.from_numpy(obs).to(dtype=torch.float32)}
            a_dict = module.forward_inference(obs_dict)
            if "actions" in a_dict:
                act = a_dict["actions"].cpu().numpy()
            else:
                act_logits = a_dict["action_dist_inputs"]
                action_dist_cls = module.get_inference_action_dist_cls()
                act_dist = action_dist_cls.from_logits(act_logits)
                act = act_dist.sample().cpu().numpy()

            # env step
            obs, rew, term, trunc, info = env.step(act)
            done = done | term | trunc
            tot += rew * (~done).astype(np.float32)
        data[ep_ind * nenvs : (ep_ind + 1) * nenvs] = tot
    return data


@ray.remote
def evaluate_policy(
    path: str,
    env_str: str,
    context_ind: int,
    nmesh: int,
    vectorization: int,
    ray_pbar: tqdm_ray.tqdm,
):

    with torch.no_grad():
        rl_module = RLModule.from_checkpoint(path).compile(TorchCompileConfig())

        env_cls = env_class_dict[env_str]
        tmp_args = SimpleNamespace(cse=0, env=env_str, solver="euler")
        base_context = get_env_config(tmp_args)["context_values"]
        time_limit = time_limit_dict[env_str]
        env = SyncVectorEnv(
            [
                lambda p=base_context: TimeLimit(env_cls(p), time_limit)
                for _ in range(vectorization)
            ]
        )

        low = env_cls.context_low
        high = env_cls.context_high

        if isinstance(low, np.ndarray):
            assert low.shape == base_context.shape
            low = low[context_ind]
        if isinstance(high, np.ndarray):
            assert high.shape == base_context.shape
            high = high[context_ind]

        if env_str in ["ant_dir", "ant_goal"]:
            x = np.linspace(0, 2 * np.pi, nmesh)
            x_coord = np.cos(x)
            y_coord = np.sin(x)
            new_contexts = np.stack([x_coord, y_coord], axis=1)
        else:
            x = np.linspace(low, high, nmesh)
            new_contexts = np.repeat(base_context.reshape(1, -1), repeats=nmesh, axis=0)
            new_contexts[:, context_ind] = x

        returns = np.zeros((new_contexts.shape[0], NEPISODES))

        # evaluate on a bunch of perturbations
        for i, new_context in enumerate(new_contexts):
            env.call("update_context", new_context)

            # evaluate the policy
            returns[i, :] = get_returns(env, rl_module, nepisodes=NEPISODES)
            ray_pbar.update.remote(1)

    env.close()

    return dict(
        context_mesh=x,
        new_contexts=new_contexts,
        returns=returns,
        base_context=base_context,
    )


if __name__ == "__main__":
    args = make_plot_args()

    path_start = os.path.join(os.getcwd(), "experiments")
    MAX_CONCURRENT = 6

    # Need to pass in a bunch of numpy arrays for the arguments.

    path_list = []
    env_str_list = []
    context_ind_list = []
    save_path_list = []
    print("Computing for the following specs:")
    for e in args.env:
        if e[:3] == "bc_":
            method_list = ["bc"]
        else:
            method_list = ["baseline", "cse1", "dr1"]
        for method in method_list:
            # Get experiment path
            if method == "bc":
                env_str = e[3:]
            else:
                env_str = e
            exp_path = os.path.join(path_start, e, f"{env_str}_{method}")
            if not os.path.exists(exp_path):
                continue

            # Get default context
            env_args = SimpleNamespace(env=env_str, cse=args.cse, solver=args.solver)
            train_context = get_env_config(env_args)["context_values"]

            # * Set up save paths
            config_str = (
                "("
                + "_".join([str(x) for x in train_context])
                + ")"
                + f"_{args.nmesh}_{args.checkpoint_num}"
            )

            policy_paths = get_policy_paths(exp_path, args.checkpoint_num)

            print("\t", e, method)

            # make the directory if it doesn't exist
            if not os.path.exists(os.path.join(path_start, e, "processed")):
                os.makedirs(os.path.join(path_start, e, "processed"))
            save_path = os.path.join(path_start, e, "processed", config_str)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Will save separately for each model

            if e in ["ant_dir", "ant_goal"]:
                num_cind = 1
            else:
                num_cind = len(train_context)

            for cind in range(num_cind):
                for pind, p in enumerate(policy_paths):
                    # If the save path for this policy already exists, skip it
                    specific_save_path = os.path.join(
                        save_path, f"{method}_pind{pind}_cind{cind}.npz"
                    )
                    if os.path.exists(specific_save_path):
                        # If a skip occurs, the file already exists
                        print("\t\t", f"Model {pind}", f"cind {cind}\t (skipped)")
                        continue

                    # Add data to lists
                    path_list.append(p)
                    env_str_list.append(env_str)
                    context_ind_list.append(cind)
                    save_path_list.append(specific_save_path)
                    print("\t\t", f"Model {pind}", f"cind {cind}")
    print("Total number of policies to evaluate:", len(path_list))
    if len(path_list) == 0:
        exit()
    length_dict = {
        "path_list": len(path_list),
        "env_str_list": len(env_str_list),
        "context_ind_list": len(context_ind_list),
        "save_path_list": len(save_path_list),
    }
    if len(set(length_dict.values())) != 1:
        raise ValueError(
            f"Length of lists are not equal: {length_dict}. Please check the lists."
        )

    #! HERE
    ray.init(num_cpus=MAX_CONCURRENT)
    remote_tqdm = ray.remote(tqdm_ray.tqdm)

    pbar = remote_tqdm.remote(total=len(save_path_list) * args.nmesh)

    data_comp = []
    for pa, ec, ci in zip(path_list, env_str_list, context_ind_list):
        data_comp.append(
            evaluate_policy.remote(
                pa,
                ec,
                ci,
                args.nmesh,
                args.vectorization,
                pbar,
            )
        )
    data = ray.get(data_comp)
    pbar.close.remote()

    # Let the pbar finish
    time.sleep(0.2)
    ray.shutdown()

    # Loop and save the data
    for d, sp in zip(data, save_path_list):
        np.savez(os.path.join(sp), **d)
    print("Done")
