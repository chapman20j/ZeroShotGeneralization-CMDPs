# dr.py
"""
This file implements domain randomization:
- local
- global

This is implemented as an RLlib callback.
"""
from typing import Optional

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner


# * Domain Randomization Callbacks
class LocalDomainRandomizationCallback(RLlibCallback):

    def __init__(self, base_context: np.ndarray, delta: float, default_eval: bool):
        super().__init__()
        self.base_context = base_context
        self.delta = delta
        self.default_eval = default_eval

    def on_episode_created(
        self,
        *,
        env_runner: Optional[SingleAgentEnvRunner] = None,
        env: Optional[gym.vector.SyncVectorEnv] = None,
        **kwargs,
    ) -> None:
        # NOTE: We only do DR if we are not in evaluation mode
        # this works because the evaluation_config for the eval_env_runner is None
        if env_runner.config.evaluation_config:
            env.call("update_context_gaussian", self.base_context, self.delta)

    def on_evaluate_start(self, *, algorithm: Algorithm, **kwargs) -> None:
        # resets the context for evaluation
        # this is important when not using --custom_eval
        if self.default_eval:
            algorithm.env_runner.env.env.call("update_context", self.base_context)


class GlobalDomainRandomizationCallback(RLlibCallback):

    def __init__(
        self,
        base_context: np.ndarray,
        context_low: float | np.ndarray,
        context_high: float | np.ndarray,
        default_eval: bool,
    ):
        super().__init__()
        self.base_context = base_context
        self.low = context_low
        self.high = context_high
        self.default_eval = default_eval

    def on_episode_created(
        self,
        *,
        env_runner: Optional["EnvRunner"] = None,
        env: Optional[gym.vector.SyncVectorEnv] = None,
        **kwargs,
    ) -> None:
        # NOTE: We only do DR if we are not in evaluation mode
        # this works because the evaluation_config for the eval_env_runner is None
        if env_runner.config.evaluation_config:
            env.call("update_context_uniform", self.low, self.high)

    def on_evaluate_start(self, *, algorithm: Algorithm, **kwargs) -> None:
        # resets the context for evaluation
        # this is important when not using --custom_eval
        if self.default_eval:
            algorithm.env_runner.env.env.call("update_context", self.base_context)
