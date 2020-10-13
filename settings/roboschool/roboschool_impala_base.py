from .roboschool_base import *
import sys

network_opts["entropy_beta"] = 0.01

workers = 8
worker_opts.update({
    "norm_clip": None,
    "opt_epochs": 1,
    "batch_size": 2,
    "unroll_length": 32
})

max_samples = int(sys.argv[sys.argv.index("--max_samples")+1]) if "--max_samples" in sys.argv else 3000000
max_iterations = 10000000

model = "LearnerModel"
worker = "IMPALAWorker"
