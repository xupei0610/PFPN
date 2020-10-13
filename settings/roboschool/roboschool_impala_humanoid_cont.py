from .humanoid_impala_humanoid_base import *

worker_opts["lr_critic"] = 1e-4
worker_opts["lr_actor"] = 3e-5

network = "ContinuousVTraceNetwork"
