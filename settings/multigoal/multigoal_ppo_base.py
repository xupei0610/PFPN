import tensorflow as tf
import numpy as np
import sys

network_opts = {
    "common_net_shape": [],
    "critic_net_shape": [128, 128],
    "actor_net_shape": [128, 128],
    "weight_initializer": tf.constant_initializer,
    "activator": tf.nn.relu,
    "value_loss_coef": 0.5,
    "gamma": 0.9,
    "normalize_advantage": True, 

    "lambd": 0.95
}


workers = 1
worker_opts = {
    "lr_critic": 3e-4,
    "lr_actor":  3e-4,
    "norm_clip": 1.0,
    "opt_epochs": 1,
    "batch_size": 64,
    "unroll_length": 512
}

_ceil = lambda x, y: -(-x // y)
_max_samples = int(sys.argv[sys.argv.index("--max_samples")+1]) if "--max_samples" in sys.argv else 1000000
max_iterations = _ceil(_max_samples, worker_opts["unroll_length"]*workers) * worker_opts["unroll_length"]/worker_opts["batch_size"]*worker_opts["opt_epochs"]

model = "SyncModel"
worker = "DPPOWorker"
