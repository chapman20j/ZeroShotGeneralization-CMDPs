# ant_goal.py
"""
Trains an ant to run in a direction specified by the context.
This is based on
    https://arxiv.org/pdf/2105.13524
    https://github.com/suyoung-lee/LDM/blob/main/environments/mujoco/ant_goal.py
"""


import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.spaces import Box


class AntGoal(AntEnv):
    context_low = np.array([-1.0, -1.0])
    context_high = np.array([1.0, 1.0])

    def __init__(
        self,
        context_values: np.ndarray,
        *args,
        healthy_reward: float = 0.0,
        terminate_when_unhealthy: bool = False,
        **kwargs,
    ):
        assert context_values.shape == (
            2,
        ), f"Context should be of shape (2,), but got: {context_values.shape}"
        kwargs["healthy_reward"] = healthy_reward
        kwargs["terminate_when_unhealthy"] = terminate_when_unhealthy
        kwargs["exclude_current_positions_from_observation"] = False
        super().__init__(*args, **kwargs)

        obs_size = self.observation_space.shape[0]
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size + 2,), dtype=np.float64
        )

        self.context_values = context_values
        self.context_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(xy_position_after, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        o = self.observation_space.shape[0] - 2
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": xy_velocity[0],
            "y_velocity": xy_velocity[0],
            **reward_info,
            "dobs_dcontext": np.zeros((o, 2)),
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_reset_info(self):
        o = self.observation_space.shape[0] - 2
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "dobs_dcontext": np.zeros((o, 2)),
            "dr_dcontext": np.zeros((2,)),
            "dr_dnextobs": np.zeros(o),
        }

    def _get_rew(self, xy_position: np.ndarray, action):
        diff = xy_position - self.context_values
        tmp = np.sqrt(1 + np.linalg.norm(diff) ** 2)
        goal_reward = 1 - tmp
        healthy_reward = self.healthy_reward
        rewards = goal_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        drdp = diff / tmp
        o = self.observation_space.shape[0] - 2
        reward_info = {
            "reward_goal": goal_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "dr_dcontext": drdp,
            "dr_dnextobs": np.concatenate([-drdp, np.zeros(o - 2)]),
        }

        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate(
                (position, velocity, contact_force, self.context_values)
            )
        else:
            return np.concatenate((position, velocity, self.context_values))

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
