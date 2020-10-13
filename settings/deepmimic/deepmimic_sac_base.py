from .deepmimic_base import *
import sys

workers = 1
worker_opts.update({
    "norm_clip": 1,
    "opt_epochs": None,
    "batch_size": 256,
    "unroll_length": 1,
    "buffer_capacity": int(1e6),
    "observations": 0
})


max_iterations = int(sys.argv[sys.argv.index("--max_samples")+1]) if "--max_samples" in sys.argv else 15000000
worker = "SyncSACWorker"
model = "SyncModel"
