
from .deepmimic_impala_base import *
from ..wrappers import DiscreteActionWrapper
import sys

network_opts["particles"] = int(sys.argv[sys.argv.index("--particles")+1])
worker_opts["lr_actor"] = 1e-4

env_wrapper = lambda e: DiscreteActionWrapper(e, network_opts["particles"])

network = "DiscreteVTraceNetwork"

