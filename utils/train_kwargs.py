# train_kwargs.py


# Mine
def get_simple_config():
    return dict(
        gamma=0.9,
        actor_lr=1e-3,
        critic_lr=2e-3,
        alpha_lr=4e-4,
        num_steps_sampled_before_learning_starts=256,
    )


def get_simple_dir_config():
    return dict(
        gamma=0.9,
        actor_lr=1e-3,
        critic_lr=2e-3,
        alpha_lr=4e-4,
        num_steps_sampled_before_learning_starts=256,
    )


def get_pendulum_config():
    return dict(
        initial_alpha=1.001,
        # Use a smaller learning rate for the policy.
        actor_lr=2e-4,
        critic_lr=8e-4,
        alpha_lr=9e-4,
        lr=None,
        target_entropy="auto",
        n_step=1,
        tau=0.005,
        train_batch_size_per_learner=256,
        target_network_update_freq=1,
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 100000,
            "alpha": 1.0,
            "beta": 0.0,
        },
        num_steps_sampled_before_learning_starts=256,
    )


def get_cartpole_config():
    return dict(
        lr=5e-4,
        train_batch_size_per_learner=32,
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 50000,
            "alpha": 0.6,
            "beta": 0.4,
        },
        n_step=1,
        double_q=True,
        dueling=True,
        epsilon=[(0, 1.0), (10000, 0.02)],
    )


# Lets try this
def get_cheetah_config():
    return dict(
        initial_alpha=1.001,
        actor_lr=2e-4,
        critic_lr=8e-4,
        alpha_lr=9e-4,
        lr=None,
        target_entropy="auto",
        n_step=1,
        tau=0.005,
        train_batch_size_per_learner=256,
        target_network_update_freq=1,
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 100000,
            "alpha": 0.6,
            "beta": 0.4,
        },
        num_steps_sampled_before_learning_starts=10000,
    )


def get_ant_config():
    # https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/sac/benchmark_sac_mujoco.py
    return dict(
        initial_alpha=1.001,
        # Choose a smaller learning rate for the actor (policy).
        actor_lr=3e-5,
        critic_lr=3e-4,
        alpha_lr=1e-4,
        target_entropy="auto",
        n_step=1,
        tau=0.005,
        train_batch_size=256,
        target_network_update_freq=1,
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 1000000,
            "alpha": 0.6,
            "beta": 0.4,
        },
        num_steps_sampled_before_learning_starts=256,
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "post_fcnet_weights_initializer": "orthogonal_",
            "post_fcnet_weights_initializer_config": {"gain": 0.01},
        },
    )


def get_train_config(name):
    if name == "simple_dir":
        return get_simple_dir_config()
    elif name in ["pen", "pen_goal", "Pendulum-v1"]:
        return get_pendulum_config()
    elif name in ["cart_goal", "CartPole-v1"]:
        return get_cartpole_config()
    elif name in ["ant_dir", "ant_goal"]:
        return get_ant_config()
    elif name == "cheetah_vel":
        return get_cheetah_config()
    else:
        raise ValueError(f"Environment {name} not recognized for training kwargs.")
