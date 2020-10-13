from .roboschool_base import *
import sys

network_opts["normalize_advantage"] = True

workers = 1
worker_opts.update({
    "norm_clip": 1.0,
    "opt_epochs": 10,
    "batch_size": 128/workers,
    "unroll_length": 2048/workers,
})

_ceil = lambda x, y: -(-x // y)
_max_samples = int(sys.argv[sys.argv.index("--max_samples")+1]) if "--max_samples" in sys.argv else 3000000
max_iterations = _ceil(_max_samples, worker_opts["unroll_length"]*workers) * worker_opts["unroll_length"]/worker_opts["batch_size"]*worker_opts["opt_epochs"]

model = "SyncModel"
worker = "DPPOWorker"
