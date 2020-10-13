from .roboschool_a2c_base import *
from ..wrappers import DiscreteActionWrapper
import sys

network_opts["particles"] = int(sys.argv[sys.argv.index("--particles")+1])
env_wrapper = lambda e: DiscreteActionWrapper(e, network_opts["particles"])

network = "DiscreteA2CNetwork"
