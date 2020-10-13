import tensorflow as tf
import numpy as np

network_opts = {
    "common_net_shape": [],
    "critic_net_shape": [256, 256],
    "actor_net_shape": [256, 256],
    "weight_initializer": tf.orthogonal_initializer,
    #tf.glorot_uniform_initializer,
    "activator": tf.nn.relu,
    "value_loss_coef": 0.5,

    "gamma": 0.99,

    "normalize_state": False,
    "clip_state": None,
    "normalize_value": False,
    "clip_value": None,
    "normalize_advantage": False,
    "clip_advantage": None,

    # "init_sigma": 1.,
    # "variabilize_sigma": True
}

worker_opts = {
    "lr_critic": 3e-4,
    "lr_actor": 3e-4
}
