# common.py

from argparse import Namespace

import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv, PendulumEnv
from gymnasium.wrappers import TimeLimit

from environments.mujoco_context_envs.ant_dir import AntDir
from environments.mujoco_context_envs.ant_goal import AntGoal
from environments.mujoco_context_envs.cheetah_velocity import HalfcheetahVelocity
from environments.ode_env import CartPoleGoalAnalytic, PendulumGoalAnalytic
from environments.simple import SimpleDirection

env_class_dict = {
    "pen_goal": PendulumGoalAnalytic,
    "cart_goal": CartPoleGoalAnalytic,
    "simple_dir": SimpleDirection,
    "ant_dir": AntDir,
    "ant_goal": AntGoal,
    "cheetah_vel": HalfcheetahVelocity,
    "CartPole-v1": CartPoleEnv,
    "Pendulum-v1": PendulumEnv,
}
time_limit_dict = {
    "simple_dir": 10,
    "pen_goal": 200,
    "cart_goal": 500,
    "ant_dir": 1000,
    "ant_goal": 1000,
    "cheetah_vel": 1000,
    "Pendulum-v1": 200,
    "CartPole-v1": 500,
}


def env_lambda(name):
    time_limit = time_limit_dict.get(name, 1000)
    if name in ["pen_goal", "cart_goal"]:
        return lambda config: TimeLimit(
            env_class_dict[name](
                config["context_values"],
                solver=config["solver"],
            ),
            time_limit,
        )
    if name in ["ant_dir", "ant_goal", "cheetah_vel"]:
        return lambda config: TimeLimit(
            env_class_dict[name](
                config["context_values"],
            ),
            time_limit,
        )
    if name in ["CartPole-v1", "Pendulum-v1"]:
        return lambda config: TimeLimit(
            env_class_dict[name](),
            time_limit_dict[name],
        )
    return lambda config: TimeLimit(
        env_class_dict[name](
            config["context_values"],
        ),
        time_limit,
    )


def get_env_config(input_args: Namespace) -> dict:
    other_info = True
    if input_args.env == "cart_goal":
        env_config = {
            "context_values": np.array([10, 0.1, 1, 0.5, 0]),
            "perturbations": np.array(
                [
                    [10, 0.1, 1, 0.5, 0],
                    [10, 0.1, 1, 0.5, -0.5],
                    [10, 0.1, 1, 0.5, 0.5],
                    [10, 0.1, 1, 0.5, -1],
                    [10, 0.1, 1, 0.5, 1],
                ]
            ),
        }
    elif input_args.env == "pen_goal":
        env_config = {
            "context_values": np.array([2.0, 1.0, 1.0, 0.0]),
            "perturbations": np.array(
                [
                    [2.0, 1.0, 1.0, 0.0],
                    [2.0, 1.0, 1.0, 0.75],
                    [2.0, 1.2, 1.0, -0.75],
                ]
            ),
        }
    elif input_args.env == "simple_dir":
        tmp = np.array([[0, 0], [0.8, 0.8], [-0.8, 0.8], [-0.8, -0.8], [0.8, -0.8]])
        env_config = {
            "context_values": tmp[0],
            "perturbations": tmp,
        }
    elif input_args.env == "ant_dir":
        th = np.array([0, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5, 1, -1, np.pi])
        tmp = np.stack([np.cos(th), np.sin(th)], axis=-1)
        env_config = {
            "name": input_args.env,
            "context_values": tmp[0],
            "perturbations": tmp,
        }
    elif input_args.env == "ant_goal":
        th = np.array([0, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5, 1, -1, np.pi])
        tmp = 3 * np.stack([np.cos(th), np.sin(th)], axis=-1)
        env_config = {
            "name": input_args.env,
            "context_values": tmp[0],
            "perturbations": tmp,
        }
    elif input_args.env == "cheetah_vel":

        env_config = {
            "name": input_args.env,
            "context_values": np.array([2.0]),
            "perturbations": np.array(
                np.array([[2.0], [0.75], [1.25], [1.75], [2.25], [2.75], [-1.0]])
            ),
        }
    else:
        env_config = dict()
        other_info = False

    if other_info:
        env_config["solver"] = input_args.solver

    return env_config
