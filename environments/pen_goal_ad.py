# pen_goal_ad.py

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces


@jax.jit
def pengoal_fwd(state, context_values, action, dt):
    # theta'' = (3g/(2l)) * sin(theta) + 3 u / (m l^2)
    g, m, l, tau = context_values
    u = action[0]
    theta = state[0]
    dtheta_dt = state[1]
    eqn = (3 * g) / (2 * l) * jnp.sin(theta) + 3 * u / (m * l * l)

    return jnp.array([theta + dt * dtheta_dt, dtheta_dt + dt * eqn])


@jax.jit
def to_observables(state, context_values):
    theta = state[0]
    dtheta_dt = state[1]
    return jnp.array([jnp.cos(theta), jnp.sin(theta), dtheta_dt])


@jax.jit
def full_fwd(state, context_values, action, dt):
    next_state = pengoal_fwd(state, context_values, action, dt)
    return to_observables(next_state, context_values)


dobs_dcontext_fn = jax.jit(jax.jacrev(full_fwd, argnums=1))


@jax.jit
def pengoal_reward(state, context_values, action):
    g, m, l, tau = context_values
    goal = jnp.asin(-tau / (m * g * l))
    reward_expr = -(
        np.pi**2 * jnp.sin((state[0] - goal) / 2) ** 2
        + 0.1 * state[1] ** 2
        + 0.001 * action**2
    )
    return reward_expr


dr_fn = jax.jit(jax.jacrev(pengoal_reward, argnums=(0, 1)))


def full_dynamics(state, context_values, action, dt):
    # Full dynamics function that includes the conversion to observables
    next_state = pengoal_fwd(state, context_values, action, dt)
    next_obs = to_observables(next_state, context_values)
    dobs_dcontext = dobs_dcontext_fn(next_state, context_values, action, dt)
    reward = pengoal_reward(state, context_values, action)
    dr_dnextstate, dr_dcontext = dr_fn(state, context_values, action)

    dr_dtheta = dr_dnextstate[0][0]
    dr_ddtheta_dt = dr_dnextstate[0][1]
    dr_dcostheta = dr_dtheta / (-jnp.sin(next_state[0]))
    dr_dsin_theta = dr_dtheta / jnp.cos(next_state[0])
    dr_dnextobs = jnp.array([dr_dcostheta, dr_dsin_theta, dr_ddtheta_dt])

    return (
        next_state,
        np.array(jnp.concat([next_obs, context_values])),
        np.array(dobs_dcontext),
        reward[0],
        np.array(dr_dcontext[0]),
        np.array(dr_dnextobs),
    )


class PendulumGoalAD(gym.Env):
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
        super(PendulumGoalAD, self).__init__()
        # Context is (g, m, l, goal torque)
        assert context_values.shape == (4,)

        # * Set parameters
        self.state = None

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
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float64)

        self.append_context = append_context

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

    def step(self, action):
        (
            next_state,
            next_obs,
            dobs_dcontext,
            reward,
            dr_dcontext,
            dr_dnextobs,
        ) = full_dynamics(self.state, self.context_values, action, self.dt)
        self.state = next_state

        # Normalize the angle
        next_obs[0] = (next_obs[0] + np.pi) % (2 * np.pi) - np.pi

        info = dict(
            dobs_dcontext=dobs_dcontext,
            dr_dcontext=np.array(dr_dcontext),
            dr_dnextobs=dr_dnextobs,
        )

        return next_obs, reward, False, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([np.pi, 1.0])
        low = -high
        self.state = self.np_random.uniform(low=low, high=high, size=(2,))
        self.state = jnp.array(self.state)

        obs = to_observables(self.state, self.context_values)

        info = dict(
            dobs_dcontext=np.zeros((3, 4)),
            dr_dcontext=np.zeros(len(self.context_values)),
            dr_dnextobs=np.zeros(len(obs)),
        )
        obs = np.concatenate([obs, self.context_values])
        return obs, info

    def render(self):
        pass

    def close(self):
        pass

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
