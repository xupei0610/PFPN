from .deepmimic_dppo_base import *
import sys

network_opts["particles"] = int(sys.argv[sys.argv.index("--particles")+1])
network_opts["resample"] = -1
# network_opts["resample_interval"] = int(15000*worker_opts["opt_epochs"]/worker_opts["batch_size"])


resample_interval = 20
episode_length = 600
iterations_per_roll = worker_opts["unroll_length"]/worker_opts["batch_size"]*worker_opts["opt_epochs"]
unroll_length = worker_opts["unroll_length"] #*workers
rolls = round(episode_length*resample_interval / unroll_length)
if "--soft" in sys.argv:
    network_opts["resample_interval"] = int(rolls * worker_opts["unroll_length"])
else:
    network_opts["resample_interval"] = int(rolls * iterations_per_roll)

worker_opts["lr_actor"] = 1e-4

network = "ParticleFilteringClipPPONetwork"
