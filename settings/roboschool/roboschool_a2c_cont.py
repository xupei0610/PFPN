from .roboschool_a2c_base import *

worker_opts["lr_critic"] = 1e-4
worker_opts["lr_actor"] = 1e-4

network = "ContinuousA2CNetwork"
