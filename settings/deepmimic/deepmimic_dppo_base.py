from .deepmimic_base import *
import sys

network_opts["normalize_advantage"] = True

workers = 8
worker_opts.update({
    "norm_clip": 1.,
    "opt_epochs": 1,
    "batch_size": 32,
    "unroll_length": 512, 
})

_ceil = lambda x, y: -(-x // y)
_max_samples = int(sys.argv[sys.argv.index("--max_samples")+1]) if "--max_samples" in sys.argv else 15000000
max_iterations = _ceil(_max_samples, worker_opts["unroll_length"]*workers) * worker_opts["unroll_length"]/worker_opts["batch_size"]*worker_opts["opt_epochs"]

model = "SyncModel"
worker = "DPPOWorker"
