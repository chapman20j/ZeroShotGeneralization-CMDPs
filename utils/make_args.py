# make_args.py
"""
Creates argparser for the rllib_custom file.
"""
import argparse

from environments.common import env_class_dict


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLlib Custom")

    # * Main args
    parser.add_argument("--epochs", type=int, default=150)

    # * Experiment args
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument(
        "--sample_parallel",
        default=False,
        action="store_true",
        help="Use parallel sampling",
    )
    parser.add_argument(
        "--checkpoint_at_eval",
        default=False,
        action="store_true",
        help="Checkpoint models and training state at eval time.",
    )

    # * Env args
    parser.add_argument("--env", type=str, choices=list(env_class_dict), required=True)
    parser.add_argument(
        "--vectorization", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=["euler", "rk45", "solve_ivp"],
        help="ODE solver to use",
    )

    # * Algorithm args
    parser.add_argument(
        "--algo", type=str, default="SAC", choices=["SAC", "PPO", "DQN"]
    )

    # * CSE and DR args
    parser.add_argument(
        "--cse",
        type=int,
        help="Enable context sample enhancement. 0 for none. 1 for first ordre. 2 for second order",
        default=0,
    )
    parser.add_argument(
        "--se_action",
        type=str,
        default="none",
        choices=["none", "random", "transport", "resample"],
    )
    parser.add_argument(
        "--se_delta", type=float, default=0.1, help="Delta value for sample enhancement"
    )
    parser.add_argument(
        "--dr",
        type=int,
        default=0,
        help="Enable domain randomization. 0 for none. 1 for local. 2 for global",
    )

    # * Eval args
    parser.add_argument(
        "--custom_eval",
        default=False,
        action="store_true",
        help="Enable custom evaluation",
    )
    parser.add_argument(
        "--eval_parallel",
        default=False,
        action="store_true",
        help="Enable parallel evaluation",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        help="Interval for evaluation (in epochs)",
        default=0,
    )
    parser.add_argument(
        "--eval_iters", type=int, default=8, help="Number of evaluation iterations"
    )

    return parser.parse_args()
