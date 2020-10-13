import tensorflow as tf

network_opts = {
    "common_net_shape": [],
    "critic_net_shape": [1024, 512],
    "actor_net_shape": [1024, 512],
    "weight_initializer": lambda : tf.truncated_normal_initializer(0.0, 0.01),
    "activator": tf.nn.relu6,
    "value_loss_coef": 0.5,

    "gamma": 0.95,

    "normalize_state": True,
    "clip_state": 5.,
    "normalize_value": False,
    "clip_value": None,
    "normalize_advantage": False,
    "clip_advantage": None,
}

worker_opts = {
    "lr_critic": 1e-4,
    "lr_actor": 5e-6
}
