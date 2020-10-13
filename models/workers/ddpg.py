from .base_worker import SyncActorCriticWorker
from .base_worker import AsyncActorCriticWorker
from .base_worker import BaseActorCriticLearner

__all__ = [
    "SyncDDPGWorker",
    "AsyncDDPGWorker",
    "DDPGLearner"
]

class Buffer(list):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = [None] * self.capacity
        self.clear()

    def clear(self):
        self.pointer = 0
        self.size = 0

    def append(self, item):
        self.data[self.pointer] = item
        self.pointer = (self.pointer+1) % self.capacity
        self.size = min(self.capacity, self.size+1)

    def __len__(self):
        return self.size

def ddpg_worker_wrapper(base_actor_critic_worker):
    class DDPGWorker(base_actor_critic_worker):
        def __init__(self, buffer_capacity=int(1e6), observations=0, **kwargs):
            kwargs["_on_policy"] = False
            super().__init__(**kwargs)
            self.buffer_capacity = buffer_capacity
            self.observations = observations
            if "opt_epochs" not in kwargs:
                self.opt_epochs = None # train with steps same to unroll length

        def init(self):
            assert(not self.episodic_training)
            self.separate_optimizer = True
            super().init()

        @property
        def exp_buffers(self):
            return ["state", "action", "reward", "not_terminal", "state_"]
    
        @property
        def exp_tensors(self):
            return {
                "state": self.target_net.state,
                "action": self.target_net.action_hist,
                "reward": self.target_net.reward,
                "not_terminal": self.target_net.not_terminal,
                "state_": self.target_net.target_net.state,
            }

        @property
        def train_args(self):
            args = [
                self.exp["state"], self.exp["action"], self.exp["reward"], self.exp["not_terminal"], self.exp["state_"]
            ]
            return args

        def sample_action(self, sess, state):
            if not self.is_evaluator and len(self.exp["state"]) < self.observations:
                return [self.env.action_space.sample()]
            return super().sample_action(sess, state)

        def interact_with_env(self, sess, state, action):
            state_, reward, terminal, info = super().interact_with_env(sess, state, action)
            self.exp["not_terminal"].append(0.0 if terminal and not self.overtime(info) else 1.0)
            self.exp["state_"].append(state_)
            if len(self.exp["state"]) == self.observations:
                terminal = True
            return state_, reward, terminal, info
        
        def need_train(self, sess, terminal, last_state, info):
            if not self.is_evaluator and len(self.exp["state"]) <= self.observations:
                if len(self.exp["state"]) == self.observations:
                    if hasattr(self, "enqueue"):
                        self.observations = 0
                        return True
                    self.after_train(sess)
                return False
            else:
                return super().need_train(sess, terminal, last_state, info)

        def setup_buffers(self):
            if hasattr(self, "is_learner") and not self.is_learner: # actor
                super().setup_buffers()
            else:
                self.exp = {k: Buffer(self.buffer_capacity) for k in self.exp_buffers}
    return DDPGWorker


SyncDDPGWorker = ddpg_worker_wrapper(SyncActorCriticWorker)
AsyncDDPGWorker = ddpg_worker_wrapper(AsyncActorCriticWorker)
DDPGLearner = ddpg_worker_wrapper(BaseActorCriticLearner)
