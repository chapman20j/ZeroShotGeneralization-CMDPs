# plot.py
"""
This plots the results of evaluate_policy.py.
"""

import argparse
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environments.common import env_class_dict, get_env_config

TICK_SIZE: int = 14
FONT_SIZE: int = 16

# 1.0 is opaque, 0.0 is transparent
LEGEND_TRANSPARENCY: float | None = 1.0

# compute the average of the returns within each model
# Then compute the average and sem across models
AVE_WITHIN_MODEL: bool = True


METHOD_LABELS: dict[str, str] = {
    "baseline": lambda x: x,
    "cse1": lambda x: x + " + CSE",
    "dr1": lambda x: x + " + LDR",
}


def collect_results(
    path_list: list[str], average_within_model: bool
) -> tuple[np.ndarray, np.ndarray]:
    context_mesh = None
    returns_data = []
    for path in path_list:
        data = np.load(path, allow_pickle=True)
        if context_mesh is None:
            context_mesh = data["context_mesh"]
        rd = data["returns"]
        if average_within_model:
            rd = np.mean(rd, axis=1, keepdims=True)

        returns_data.append(rd)
    returns_data = np.concatenate(returns_data, axis=1)
    return context_mesh, returns_data


def plot_results(
    path_dict: dict[str, list[str]],
    low: float,
    high: float,
    train_context: float,
    average_within_model: bool,
    train_algo: str,
    save_name: str | None,
):

    method_list = sorted(list(path_dict.keys()))
    for method in method_list:
        cmesh, returns_data = collect_results(path_dict[method], average_within_model)
        return_pd = pd.DataFrame(returns_data)

        mean_data = return_pd.mean(axis=1)
        sem_data = return_pd.sem(axis=1)

        plt.plot(cmesh, mean_data, label=METHOD_LABELS[method](train_algo))
        plt.fill_between(
            cmesh,
            mean_data - 1.96 * sem_data,
            mean_data + 1.96 * sem_data,
            alpha=0.2,
        )

    if isinstance(train_context, list):
        for i, tc in enumerate(train_context):
            plt.axvline(
                x=tc,
                color="red",
                linestyle="--",
                label="Train" if i == 0 else None,
            )
    else:
        plt.axvline(
            x=train_context,
            color="red",
            linestyle="--",
            label="Train",
        )
    plt.xlabel("Context", fontsize=FONT_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.xlim(low, high)
    plt.ylabel("Mean Return", fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE, framealpha=LEGEND_TRANSPARENCY)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(os.path.join(os.getcwd(), "figures", f"{save_name}.png"))
    else:
        plt.show()
    plt.clf()

    return


def make_plot_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        choices=list(env_class_dict),
        help="Name of the environment to use.",
    )
    parser.add_argument(
        "--save_fig",
        action="store_true",
        default=False,
        help="Whether to save the figure or not.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main_path = os.path.join(os.getcwd(), "experiments")
    args = make_plot_args()
    main_path = os.path.join(main_path, args.env)
    main_path = os.path.join(main_path, "processed")

    env_cls = env_class_dict.get(args.env)

    folders = os.listdir(main_path)
    path = None
    for folder in folders:
        if "50" in folder:
            path = os.path.join(main_path, folder)
            break

    if path is None:
        raise ValueError("No path found with 50 in the name")
    data_paths = os.listdir(path)
    max_cind = int(sorted(data_paths)[-1].split("_")[-1][4:-4])

    path_list_dict = [dict() for _ in range(max_cind + 1)]

    for folder in os.listdir(path):
        method, model, context = folder.split("_")
        context_num = int(context[4:-4])
        if method not in path_list_dict[context_num]:
            path_list_dict[context_num][method] = []
        path_list_dict[context_num][method].append(os.path.join(path, folder))

    low = env_cls.context_low
    high = env_cls.context_high
    if isinstance(low, float):
        low = [low] * len(path_list_dict)
    if isinstance(high, float):
        high = [high] * len(path_list_dict)
    training_algorithm = "DQN" if args.env == "cart_goal" else "SAC"

    env_args = SimpleNamespace(env=args.env, cse=0, solver="euler")
    train_context = get_env_config(env_args).get("context_values")

    if args.env in ["ant_dir", "ant_goal"]:
        low = [-1e-1]
        high = [2 * np.pi + 1e-1]
        train_context = [[0.0, 2 * np.pi]]

    for cind, pdict in enumerate(path_list_dict):
        save_name = f"{args.env}_{cind}" if args.save_fig else None

        plot_results(
            pdict,
            low[cind],
            high[cind],
            train_context[cind],
            AVE_WITHIN_MODEL,
            training_algorithm,
            save_name,
        )
