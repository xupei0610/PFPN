from .deepmimic_dppo_base import *

network_opts["init_sigma"] = 0.05
network = "ContinuousClipPPONetwork"