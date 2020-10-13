from .roboschool_base import *
import sys

network_opts["entropy_beta"] = 0.01

workers = 1
worker_opts.update({
    "norm_clip": 0.5,
    "opt_epochs": 1,
    "batch_size": 32,
    "unroll_length": 32
})

_ceil = lambda x, y: -(-x // y)
_max_samples = int(sys.argv[sys.argv.index("--max_samples")+1]) if "--max_samples" in sys.argv else 3000000
max_iterations = _ceil(_max_samples, worker_opts["batch_size"])

model = "AsyncModel"
worker = "AsyncA2CWorker"
