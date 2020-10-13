from .roboschool_base import *
import sys

workers=4
worker_opts.update({
    "norm_clip": 1.,
    "opt_epochs": None,
    "batch_size": 256,
    "unroll_length": 16,
    "buffer_capacity": int(1e6),
    "observations": 0,
    "queue_batch_size": 2
})

max_iterations = int(sys.argv[sys.argv.index("--max_samples")+1]) if "--max_samples" in sys.argv else 30000000
worker = "SACLearner"
model = "LearnerModel"
