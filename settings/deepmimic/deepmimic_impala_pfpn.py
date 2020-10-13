from .deepmimic_impala_base import *
import sys

network_opts["particles"] = int(sys.argv[sys.argv.index("--particles")+1])
network_opts["resample"] = -1
network_opts["resample_interval"] = int(12000*workers/worker_opts["batch_size"]/worker_opts["unroll_length"])
worker_opts["lr_actor"] = 1e-4

network = "ParticleFilteringVTraceNetwork"

