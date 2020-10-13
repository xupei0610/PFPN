from .deepmimic_sac_async_base import *
import sys

network_opts["particles"] = int(sys.argv[sys.argv.index("--particles")+1])
network_opts["resample"] = -1
network_opts["resample_interval"] = 12000

worker_opts["lr_actor"] = 1e-4

network = "ParticleFilteringSACNetwork"
