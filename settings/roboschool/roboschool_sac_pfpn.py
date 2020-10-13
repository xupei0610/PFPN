from .roboschool_sac_base import *
import sys

network_opts["particles"] = int(sys.argv[sys.argv.index("--particles")+1])
network_opts["resample"] = -1
network_opts["resample_interval"] = 20000

network = "ParticleFilteringSACNetwork"

