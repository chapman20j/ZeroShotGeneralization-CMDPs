# cse.py
"""
This file implements sample enhancement using an RLlib connector.

Enhancing transitions:
    First order
    Exact

Enhancing rewards:
    First order
    Exact

Enhancing actions:
    Transport
    Resample
    Random
    None
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override


def make_pert(size: int):
    pert = np.random.randn(size)
    tmp = np.linalg.norm(pert)
    if size == 1:
        pert = pert.reshape(1)
    return pert / tmp


def transport_sample(
    sample1: torch.Tensor,
    loc1: torch.Tensor,
    scale1: torch.Tensor,
    loc2: torch.Tensor,
    scale2: torch.Tensor,
):
    """
    Transports a sample from one Gaussian distribution to another.
    Because we have diagonal covariance matrices, we get the simple formula
    transported_sample = loc2 + (sample1 - loc1) * (scale2 / scale1)
    Note that scale is the standard deviation, not the variance.
    Args:
        sample1: The sample from the first Gaussian distribution.
        loc1: The mean of the first Gaussian distribution.
        scale1: The standard deviation of the first Gaussian distribution.
        loc2: The mean of the second Gaussian distribution.
        scale2: The standard deviation of the second Gaussian distribution.
    Returns:
        The transported sample from the second Gaussian distribution.
    """
    return loc2 + (sample1 - loc1) * (scale2 / scale1)


# * Enhancement Functions
def enhance_transition_first_order(
    *,
    obs,
    next_obs,
    pert: dict[str, np.ndarray],
    deriv: dict[str, np.ndarray],
    delta: dict[str, float],
):
    # Returns just the state part
    s = pert["s"].shape[-1]

    # Change outputs
    next_state_change_s = 0 if delta["s"] < 1e-3 else deriv["dobs_ds"] @ pert["s"]
    next_state_change_c = deriv["dobs_dcontext"] @ pert["c"]
    next_state_change_a = 0 if delta["a"] < 1e-3 else deriv["dobs_da"] @ pert["a"]
    return (
        next_obs[:s]
        + delta["s"] * next_state_change_s
        + delta["c"] * next_state_change_c
        + delta["a"] * next_state_change_a
    )


def enhance_transition_exact(
    *,
    obs,
    act,
    pert: dict[str, np.ndarray],
    transition_fn,
):
    # Gives just the state part
    s = pert["s"].shape[-1]
    p = pert["c"].shape[-1]

    # Change outputs
    return transition_fn(obs[:s], act, obs[-p:])


@torch.no_grad()
def enhance_action_transport(
    *, obs, model, act, delta: dict[str, float], pert: dict[str, np.ndarray]
):
    obs = torch.from_numpy(obs).to(torch.float32)
    full_pert = torch.from_numpy(
        np.concatenate([pert["s"] * delta["s"], pert["c"] * delta["c"]])
    ).to(torch.float32)

    obs_pert = obs + full_pert

    # Get the base action
    obs_dict = {Columns.OBS: torch.stack((obs, obs_pert))}
    pi_enc = model.pi_encoder(obs_dict)
    action_logits = model.pi(pi_enc["encoder_out"])
    action_dists = model.get_train_action_dist_cls().from_logits(action_logits)
    loc = action_dists.loc.numpy()
    scale = action_dists._dist.scale.numpy()
    return transport_sample(act, loc[0], scale[0], loc[1], scale[1])


@torch.no_grad()
def enhance_action_resample(
    *, obs, model, delta: dict[str, float], pert: dict[str, np.ndarray]
):
    obs = torch.from_numpy(obs).to(torch.float32)
    full_pert = torch.from_numpy(
        np.concatenate([pert["s"] * delta["s"], pert["c"] * delta["c"]])
    ).to(torch.float32)

    obs_pert = obs + full_pert

    # Get the base action
    obs_dict = {Columns.OBS: obs_pert}
    enc = model.forward(obs_dict)
    action_logits = enc["action_dist_inputs"]
    action_dists = model.get_train_action_dist_cls().from_logits(action_logits)
    return action_dists.sample().numpy()


def enhance_reward_first_order(
    *,
    rew,
    pert: dict[str, np.ndarray],
    deriv: dict[str, np.ndarray],
    delta: dict[str, float],
):

    # Change outputs
    next_state_change_s = (
        np.zeros_like(pert["s"]) if delta["s"] < 1e-3 else deriv["dobs_ds"] @ pert["s"]
    )
    next_state_change_c = deriv["dobs_dcontext"] @ pert["c"]
    next_state_change_a = (
        np.zeros_like(pert["s"]) if delta["a"] < 1e-3 else deriv["dobs_da"] @ pert["a"]
    )
    drda_part = 0 if delta["a"] < 1e-3 else deriv["dr_da"] @ pert["a"]

    rew_s_lin = deriv["dr_dnextobs"] @ next_state_change_s
    rew_c_lin = (
        deriv["dr_dcontext"] @ pert["c"] + deriv["dr_dnextobs"] @ next_state_change_c
    )
    rew_a_lin = drda_part + deriv["dr_dnextobs"] @ next_state_change_a
    out_rew = (
        rew + delta["s"] * rew_s_lin + delta["c"] * rew_c_lin + delta["a"] * rew_a_lin
    )
    return out_rew


def enhance_reward_exact(state, act, next_state, context, rew_fn):
    return rew_fn(state, act, next_state, context)


# NOTE: Assumes that R(s, a, s') = R(a,s') for gradients
def enhance_single_sample(
    *,
    obs,
    act,
    new_act,
    rew,
    next_obs,
    pert: dict[str, np.ndarray],
    deriv: dict[str, np.ndarray],
    delta: dict[str, float],
    transition_fn: Optional[Callable] = None,
    rew_fn: Optional[Callable] = None,
):

    # Get action pert
    a_pert = new_act - act
    delta["a"] = np.linalg.norm(a_pert)
    if delta["a"] < 1e-2:
        pert["a"] = np.zeros_like(a_pert)
    else:
        pert["a"] = a_pert / delta["a"]

    s = pert["s"].shape[-1]
    p = pert["c"].shape[-1]

    # Enhance initial observation
    out_obs = obs.copy()
    out_obs[:s] += delta["s"] * pert["s"]
    out_obs[-p:] += delta["c"] * pert["c"]

    # Enhance next observation
    if transition_fn is not None:
        # exact transition
        out_next_state = enhance_transition_exact(
            obs=out_obs,
            act=new_act,
            pert=pert,
            transition_fn=transition_fn,
        )
    else:
        # first order transition
        out_next_state = enhance_transition_first_order(
            obs=out_obs,
            next_obs=next_obs,
            pert=pert,
            deriv=deriv,
            delta=delta,
        )

    # Append context back
    out_next_obs = np.concatenate([out_next_state, out_obs[-p:]])

    # Enhance reward
    if rew_fn is not None:
        # exact rewards
        out_rew = rew_fn(out_obs[:s], act, out_next_obs[:s], out_obs[-p:])
    else:
        # first order rewards
        out_rew = enhance_reward_first_order(
            rew=rew,
            pert=pert,
            deriv=deriv,
            delta=delta,
        )

    return out_obs, out_rew, out_next_obs


class SampleEnhanceConnector(ConnectorV2):

    def __init__(
        self,
        input_obs_space=None,
        input_act_space=None,
        transition_fn: Optional[Callable] = None,
        reward_fn: Optional[Callable] = None,
        action_mtd: str = "transport",
        delta: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(input_obs_space, input_act_space, **kwargs)
        if delta is None:
            raise ValueError("Delta must be specified")
        self.orig_delta = {k: v for k, v in delta.items()}
        self.delta = delta

        # Specify these to indicate you want to use exact rewards
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.action_mtd = action_mtd
        if action_mtd not in ["transport", "random", "none", "resample"]:
            raise ValueError(
                f"Action method {action_mtd} not supported. Use 'transport', 'random', 'none', or 'resample'."
            )

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: "RLModule",
        batch: Optional[Dict[str, Any]],
        episodes: List["EpisodeType"],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        model = rl_module.get("default_policy")

        if self.delta["a"] < 1e-8 and self.delta["c"] < 1e-8 and self.delta["s"] < 1e-8:
            return batch

        for sa_episode in self.single_agent_episode_iterator(
            episodes,
            agents_that_stepped_only=False,
        ):
            # This gets info from the next step. This is important for how we store the gradients
            info = sa_episode.get_infos(slice(1, len(sa_episode) + 1))

            # * Generate perturbations
            s, p = info[0]["dobs_dcontext"].shape
            pert = dict(s=make_pert(s), c=make_pert(p))

            # * Enhance Action
            if self.action_mtd == "random":
                a_pert = make_pert(sa_episode.actions[0].shape[-1])
                new_act = sa_episode.actions[0] + a_pert * self.delta["a"]
            elif self.action_mtd == "transport":
                new_act = enhance_action_transport(
                    obs=sa_episode.observations[0],
                    model=model,
                    act=sa_episode.actions[0],
                    delta=self.delta,
                    pert=pert,
                )
            elif self.action_mtd == "resample":
                new_act = enhance_action_resample(
                    obs=sa_episode.observations[0],
                    model=model,
                    delta=self.delta,
                    pert=pert,
                )
            else:
                new_act = sa_episode.actions[0]

            # * Enhance others
            new_obs, new_rew, new_next_obs = enhance_single_sample(
                obs=sa_episode.observations[0],
                act=sa_episode.actions[0],
                new_act=new_act,
                rew=sa_episode.rewards[0],
                next_obs=sa_episode.observations[1],
                pert=pert,
                deriv=info[0],
                delta=self.delta,
                transition_fn=self.transition_fn,
                rew_fn=self.reward_fn,
            )

            # * Set items
            for col, it in zip(
                [Columns.OBS, Columns.ACTIONS, Columns.REWARDS, Columns.NEXT_OBS],
                [new_obs, new_act, new_rew, new_next_obs],
            ):
                self.add_batch_item(
                    batch=batch,
                    column=col,
                    item_to_add=it,
                    single_agent_episode=sa_episode,
                )

        return batch
