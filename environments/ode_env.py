# ode_env.py
"""
CartGoal and PendulumGoal environments.
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from .ode_sympy import ODEContextSystem
from .ode_systems import make_cartpole_goal_eqns, make_pendulum_goal_eqns


class PendulumGoalAnalytic(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    context_low = np.array([0.5, 0.5, 0.5, -1.0])
    context_high = np.array([5.0, 2.0, 2.0, 1.0])

    def __init__(
        self,
        context_values,
        dt=0.05,
        append_context=True,
        solver: str = "euler",
        render_mode: str | None = None,
    ):
        super(PendulumGoalAnalytic, self).__init__()
        # Context is (g, m, l, goal torque)
        assert context_values.shape == (4,)

        # * Set parameters
        eqns, observables, context_vars, control_vars, reward_expr = (
            make_pendulum_goal_eqns()
        )

        self.system = ODEContextSystem(
            eqns,
            observables,
            context_vars,
            control_vars,
            reward_expr,
            solver=solver,
        )

        self.context_values = context_values
        self.dt = dt

        # * Set up the system
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 + len(context_values),),
            dtype=np.float64,
        )
        self.context_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(context_vars),), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(len(control_vars),), dtype=np.float64
        )

        self.append_context = append_context

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

    def step(self, action):
        obs, dobs_dcontext = self.system.step(self.context_values, action, self.dt)

        # Normalize the angle
        obs[0] = (obs[0] + np.pi) % (2 * np.pi) - np.pi

        reward, dr_dcontext, dr_ds = self.system.reward(self.context_values, action)

        info = dict(
            dobs_dcontext=dobs_dcontext,
            dr_dcontext=np.array(dr_dcontext),
        )

        info["dobs_dcontext"] = self._process_dobs_dcontext(obs, info["dobs_dcontext"])

        # Do this before updating obs
        dr_ds = self._process_dr_ds(obs, dr_ds)
        obs = self._process_obs(obs)

        # NOTE: We append on the context in this environment
        if self.append_context:
            obs = np.concatenate([obs, self.context_values])
        info["dr_dnextobs"] = dr_ds

        return obs, reward, False, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([np.pi, 1.0])
        low = -high
        self.state = self.np_random.uniform(low=low, high=high, size=(2,))
        self.system.state = np.zeros(
            len(self.system.first_order_state_variables), dtype=np.float64
        )
        self.system.state[:2] = self.state
        obs, dobs_dcontext = self.system._unpack_state(self.system.state)

        dobs_dcontext = self._process_dobs_dcontext(obs, dobs_dcontext)
        obs = self._process_obs(obs)

        if self.append_context:
            info = dict(
                dobs_dcontext=dobs_dcontext,
                dr_dcontext=np.zeros(len(self.context_values)),
                dr_dnextobs=np.zeros(len(obs)),
            )
            obs = np.concatenate([obs, self.context_values])
        else:
            info = dict(dobs_dcontext=dobs_dcontext)
        return obs, info

    def _process_obs(self, obs):
        return np.array([np.cos(obs[0]), np.sin(obs[0]), obs[1]])

    def _process_dr_ds(self, obs, dr_ds):
        return np.array(
            [dr_ds[0] / (-np.sin(obs[0])), dr_ds[0] / np.cos(obs[0]), dr_ds[1]]
        )

    def _process_dobs_dcontext(self, obs, dobs_dcontext):
        # cos(theta) -> -sin(theta) * dtheta

        return np.stack(
            [
                -np.sin(obs[0]) * dobs_dcontext[0, :],
                np.cos(obs[0]) * dobs_dcontext[0, :],
                dobs_dcontext[1, :],
            ]
        )

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.system.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(
            self.system.state[0] + np.pi / 2
        )
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

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

    def update_context_uniform(self, low, high):
        # NOTE: This has to modify values so that |2 tau| < mgl
        if isinstance(low, float):
            low = np.full(self.context_values.shape, low)
        else:
            assert low.shape == self.context_values.shape
        if isinstance(high, float):
            high = np.full(self.context_values.shape, high)
        else:
            assert self.context_values.shape == high.shape
        p = self.np_random.uniform(low, high)
        mgl = p[0] * p[1] * p[2]
        s = p[3]
        p[4] = self.np_random.uniform(-mgl / s, mgl / s)
        self.context_values = p
        return


class CartPoleGoalAnalytic(gym.Env):

    context_low = np.array([1.0, 0.01, 0.25, 0.1, -4])
    context_high = np.array([12.0, 2.0, 2.0, 2, 4])
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        context_values,
        dt=0.02,
        append_context=True,
        solver: str = "euler",
        render_mode: str | None = None,
    ):
        super(CartPoleGoalAnalytic, self).__init__()
        # Context is (g, mp, mc, l, goal)
        assert context_values.shape == (5,)

        # * Set parameters
        eqns, observables, context_vars, control_vars, reward_expr = (
            make_cartpole_goal_eqns()
        )

        self.system = ODEContextSystem(
            eqns,
            observables,
            context_vars,
            control_vars,
            reward_expr,
            solver=solver,
        )

        self.context_values = context_values
        self.dt = dt

        # * Set up the system
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(observables) + len(context_values),),
            dtype=np.float64,
        )
        self.context_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(context_vars),), dtype=np.float64
        )
        self.action_space = spaces.Discrete(2)

        self.append_context = append_context
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Rendering variables
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.render_mode = render_mode

    def step(self, action):
        a = np.array([10 if action == 1 else -10])
        obs, dobs_dcontext = self.system.step(self.context_values, a, self.dt)

        reward, dr_dcontext, dr_ds = self.system.reward(self.context_values, a)

        # obs[:4] = theta(t), theta~t#1(t), x(t), x~t#1(t)
        term = (
            obs[0] < -self.x_threshold
            or obs[0] > self.x_threshold
            or obs[2] < -self.theta_threshold_radians
            or obs[2] > self.theta_threshold_radians
        )
        info = dict(
            dobs_dcontext=dobs_dcontext,
            dr_dcontext=np.array(dr_dcontext),
            dr_dnextobs=dr_ds,
        )

        obs = np.concatenate([obs, self.context_values])
        return obs, reward, term, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(
            None, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.system.state = np.zeros(
            len(self.system.first_order_state_variables), dtype=np.float64
        )
        self.system.state[:4] = self.state
        obs, dobs_dcontext = self.system._unpack_state(self.system.state)

        info = dict(
            dobs_dcontext=dobs_dcontext,
            dr_dcontext=np.zeros(len(self.context_values)),
            dr_dnextobs=np.zeros(obs.shape[-1]),
        )
        obs = np.concatenate([obs, self.context_values])
        return obs, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        length = self.context_values[3]
        polewidth = 10.0
        polelen = scale * (2 * length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.system.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

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

    def update_context_uniform(self, low, high):
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
