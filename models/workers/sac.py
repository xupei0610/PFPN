from .ddpg import SyncDDPGWorker, AsyncDDPGWorker
from .ddpg import DDPGLearner#, DDPGActor

__all__ = [
    "SyncSACWorker",
    "AsyncSACWorker",
    "SACLearner"
]

SyncSACWorker = SyncDDPGWorker
AsyncSACWorker = AsyncDDPGWorker
SACLearner = DDPGLearner

