from .multigoal_ppo_base import *
import sys

network_opts["particles"] = int(sys.argv[sys.argv.index("--particles")+1])
network_opts["resample"] = -1

network_opts["resample_interval"] = 200

network = "ParticleFilteringClipPPONetwork"
