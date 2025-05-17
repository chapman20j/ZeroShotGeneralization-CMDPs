# cheetah_velocity.py
"""
Trains a halfcheetah to run at a velocity specified by the context.
This is based on
    https://arxiv.org/pdf/2105.13524
    https://github.com/suyoung-lee/LDM/blob/main/environments/mujoco/half_cheetah_vel.py
"""


import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
from gymnasium.spaces import Box


class HalfcheetahVelocity(HalfCheetahEnv):
    context_low = np.array([-3.0])
    context_high = np.array([3.0])

    def __init__(self, context_values: np.ndarray, *args, **kwargs):
        assert context_values.shape == (
            1,
        ), f"Context should be of shape (1,), but got: {context_values.shape}"
        super().__init__(*args, **kwargs)

        obs_size = self.observation_space.shape[0]
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size + 1,), dtype=np.float64
        )

        self.context_values = context_values
        self.context_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        o = self.observation_space.shape[0] - 1
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            **reward_info,
            "dobs_dcontext": np.zeros((o, 1)),
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def _get_reset_info(self):
        o = self.observation_space.shape[0] - 1
        return {
            "x_position": self.data.qpos[0],
            "dobs_dcontext": np.zeros((o, 1)),
            "dr_dcontext": np.zeros((1,)),
            "dr_dnextobs": np.zeros(o),
        }

    def _get_rew(self, x_velocity: float, action):
        diff = (x_velocity - self.context_values)[0]
        tmp = np.sqrt(1 + diff**2)
        velocity_penalty = 1 - tmp
        forward_reward = self._forward_reward_weight * velocity_penalty
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost

        drdp = np.array([diff / tmp])
        # NOTE: The observation space exludes the position
        o = self.observation_space.shape[0] - 1
        drdo = np.zeros(o)
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "dr_dcontext": drdp,
            "dr_dnextobs": drdo,
        }
        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        observation = np.concatenate(
            (observation, self.context_values.flatten())
        ).ravel()
        return observation

    def update_context(self, new_context: np.ndarray):
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
