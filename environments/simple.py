# simple.py
"""
These are a collection of relatively simple environments that have a strong dependence on context.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SimpleDirection(gym.Env):
    """This class implements the following dynamics
    T^c(s, a) = s + a + c
    R^c(s, a, s') = s' dot c
    All the values lie in the same space, but we limit actions to [-1, 1]
    """

    context_low = -1.0
    context_high = 1.0

    def __init__(self, context_values: np.ndarray):
        self.dim = context_values.shape[-1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.dim,),
            dtype=np.float64,
        )
        self.context_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=context_values.shape, dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=context_values.shape, dtype=np.float64
        )
        self.curr_timesteps = 0
        self.context_values = context_values

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_timesteps = 0
        high = np.array([1.0] * self.dim)
        low = -high

        self.state = self.np_random.uniform(low=low, high=high, size=(self.dim,))
        obs = np.concatenate([self.state, self.context_values])

        info = dict(
            dobs_ds=np.zeros((self.dim, self.dim)),
            dobs_da=np.zeros((self.dim, self.dim)),
            dobs_dcontext=np.zeros((self.dim, self.dim)),
            dr_da=np.zeros(self.dim),
            dr_dcontext=np.zeros_like(self.context_values),
            dr_dnextobs=np.zeros(self.dim),
        )
        return obs, info

    def step(self, action):
        self.state += action + self.context_values
        obs = np.concatenate([self.state, self.context_values])

        # Lets get the reward also
        reward = self.calculate_reward(None, None, self.state, self.context_values)
        info = dict(
            dobs_ds=np.eye(self.dim),
            dobs_da=np.eye(self.dim),
            dobs_dcontext=np.eye(self.dim),
            dr_da=np.zeros(self.dim),
            dr_dcontext=self.state,
            dr_dnextobs=self.context_values,
        )
        self.curr_timesteps += 1

        return obs, reward, False, False, info

    def render(self):
        return

    def close(self):
        return

    @staticmethod
    def calculate_transition(state, action, context):
        return state + action + context

    @staticmethod
    def calculate_reward(state, action, next_state, context):
        return next_state @ context

    def update_context(self, new_context):
        assert new_context.shape == self.context_values.shape
        self.context_values = new_context
        return

    def update_context_gaussian(self, base_context, scale):
        assert base_context.shape == self.context_values.shape
        self.context_values = base_context + scale * self.np_random.standard_normal(
            base_context.shape[-1]
        )
        return

    def update_context_uniform(self, low: float | np.ndarray, high: float | np.ndarray):
        if isinstance(low, float):
            low = np.full(self.context_values.shape, low)
        else:
            assert low.shape == self.context_values.shape
        if isinstance(high, float):
            high = np.full(self.context_values.shape, high)
        else:
            assert self.context_values.shape == high.shape
        self.context_values = self.np_random.uniform(low, high)
        return


class SimpleDirectionStochastic(gym.Env):
    """This class implements the following dynamics
    T^c(s, a) = s + a + c + rand
    R^c(s, a, s') = s' dot c + rand
    All the values lie in the same space, but we limit actions to [-1, 1]
    """

    context_low = -1.0
    context_high = 1.0

    def __init__(self, context_values: np.ndarray):
        self.dim = context_values.shape[-1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.dim,),
            dtype=np.float64,
        )
        self.context_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=context_values.shape, dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=context_values.shape, dtype=np.float64
        )
        self.curr_timesteps = 0
        self.context_values = context_values

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_timesteps = 0
        high = np.array([1.0] * self.dim)
        low = -high

        self.state = self.np_random.uniform(low=low, high=high, size=(self.dim,))
        obs = np.concatenate([self.state, self.context_values])

        info = dict(
            dobs_ds=np.zeros((self.dim, self.dim)),
            dobs_da=np.zeros((self.dim, self.dim)),
            dobs_dcontext=np.zeros((self.dim, self.dim)),
            dr_da=np.zeros(self.dim),
            dr_dcontext=np.zeros_like(self.context_values),
            dr_dnextobs=np.zeros(self.dim),
        )
        return obs, info

    def step(self, action):
        self.state += (
            action
            + self.context_values
            + self.np_random.standard_normal(self.state.shape) * 0.1
        )
        obs = np.concatenate([self.state, self.context_values])

        reward = (
            self.state @ self.context_values + self.np_random.standard_normal() * 0.1
        )
        info = dict(
            dobs_ds=np.eye(self.dim),
            dobs_da=np.eye(self.dim),
            dobs_dcontext=np.eye(self.dim),
            dr_da=np.zeros(self.dim),
            dr_dcontext=self.state,
            dr_dnextobs=self.context_values,
        )
        self.curr_timesteps += 1

        return obs, reward, False, False, info

    def render(self):
        return

    def close(self):
        return

    def update_context(self, new_context):
        assert new_context.shape == self.context_values.shape
        self.context_values = new_context
        return

    def update_context_gaussian(self, base_context, scale):
        assert base_context.shape == self.context_values.shape
        self.context_values = base_context + scale * self.np_random.standard_normal(
            base_context.shape[-1]
        )
        return

    def update_context_uniform(self, low: float | np.ndarray, high: float | np.ndarray):
        if isinstance(low, float):
            low = np.full(self.context_values.shape, low)
        else:
            assert low.shape == self.context_values.shape
        if isinstance(high, float):
            high = np.full(self.context_values.shape, high)
        else:
            assert self.context_values.shape == high.shape
        self.context_values = self.np_random.uniform(low, high)
        return
