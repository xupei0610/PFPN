from .roboschool_base import *
import sys

network_opts["entropy_beta"] = 0.01

workers = 32
worker_opts.update({
    "norm_clip": None,
    "opt_epochs": 1,
    "batch_size": 4,
    "unroll_length": 64
})

max_samples = int(sys.argv[sys.argv.index("--max_samples")+1]) if "--max_samples" in sys.argv else 30000000
max_iterations = 10000000

model = "LearnerModel"
worker = "IMPALAWorker"
