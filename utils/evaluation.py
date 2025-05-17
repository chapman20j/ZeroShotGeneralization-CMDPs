# evaluation.py
"""
Implements custom evaluation functions.
This allows us to test on ood contexts during training.
"""
import math
from typing import Tuple

import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS
from ray.rllib.utils.typing import ResultDict

# * Evaluation


def custom_eval_function_single_env_runner(
    algorithm: Algorithm,
    eval_env_runners: EnvRunnerGroup,
) -> Tuple[ResultDict, int, int]:

    nworkers = len(eval_env_runners.healthy_env_runner_ids())
    assert nworkers <= 1, "This function is only for single env runner"
    local_env_runner = nworkers == 0

    # Pick the smallest multiple of vector num envs that is greater than the eval iters
    eval_iters = algorithm.evaluation_config["eval_iters"]
    num_vector_envs = eval_env_runners._local_config["num_envs_per_env_runner"]
    eval_samples = math.ceil(eval_iters / num_vector_envs)

    env_config = eval_env_runners._local_config.env_config
    if "perturbations" in env_config:
        vals = env_config["perturbations"]
        nperts = vals.shape[-2]
    elif "scale" in env_config and "context_values" in env_config:
        nperts = env_config["scale"].shape[-1]
        pert = np.random.randn(nperts, env_config["context_values"].shape[-1])
        pert = pert / np.linalg.norm(pert, axis=-1, keepdims=True)
        pert *= env_config["scale"][:, None]
        vals = env_config["context_values"] + pert
    else:
        raise ValueError("No perturbations or scale found in env config")

    env_runner_metrics = [[] for _ in range(nperts)]
    sampled_episodes = []

    for p in range(nperts):
        eval_env_runners.foreach_env_runner(
            func=lambda worker, p=p: worker.env.unwrapped.call(
                "update_context", vals[p]
            )
        )

        for i in range(eval_samples):
            episodes_and_metrics_all_env_runners = eval_env_runners.foreach_env_runner(
                func=lambda worker: (worker.sample(), worker.get_metrics()),
                local_env_runner=local_env_runner,
            )

            sampled_episodes.extend(
                eps
                for eps_and_mtrcs in episodes_and_metrics_all_env_runners
                for eps in eps_and_mtrcs[0]
            )
        env_runner_metrics[p] = episodes_and_metrics_all_env_runners[0][1]

        # Reset metrics so we can reuse the env runner
        eval_env_runners.foreach_env_runner(
            func=lambda worker: worker.metrics.reset(),
            local_env_runner=local_env_runner,
        )

    # Reset the context back to base
    eval_env_runners.foreach_env_runner(
        func=lambda worker, p=p: worker.env.unwrapped.call("update_context", vals[0]),
        local_env_runner=local_env_runner,
    )

    eval_results = dict()
    for idx, mtrcs in enumerate(env_runner_metrics):
        # NOTE: This lets you see the evaluation over time
        algorithm.metrics.log_dict(
            mtrcs, key=(EVALUATION_RESULTS, ENV_RUNNER_RESULTS, str(idx))
        )
        eval_results[idx] = algorithm.metrics.reduce(
            key=(EVALUATION_RESULTS, ENV_RUNNER_RESULTS, str(idx))
        )

    # Compute env and agent steps from sampled episodes.
    env_steps = sum(eps.env_steps() for eps in sampled_episodes)
    agent_steps = sum(eps.agent_steps() for eps in sampled_episodes)

    return eval_results, env_steps, agent_steps
