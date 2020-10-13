from .base_worker import SyncActorCriticWorker
from .base_worker import AsyncActorCriticWorker
from .base_worker import BaseActorCriticLearner

__all__ = [
    "SyncVTraceWorker",
    "AsyncVTraceWorker",
    "VTraceLearner", "IMPALAWorker"
]

def vtrace_worker_wrapper(base_actor_critic_worker):
    class VTraceWorker(base_actor_critic_worker):

        def __init__(self, **kwargs): 
            kwargs["_on_policy"] = True
            self._episodic_training = False if "episodic_training" not in kwargs else kwargs["episodic_training"]
            kwargs["episodic_training"] = True
            super().__init__(**kwargs)

        @property
        def exp_buffers(self):
            return ["state", "action", "reward", "log_prob", "not_terminal"]
            
        @property
        def train_args(self):
            args = [
                self.exp["state"], self.exp["action"], self.exp["reward"], self.exp["log_prob"], self.exp["not_terminal"],
                self.sequence_length #if self._episodic_training else [sum(self.sequence_length)]
            ]
            return args

        @property
        def exp_tensors(self):
            return {
                "state": self.target_net.state,
                "action": self.target_net.action,
                "reward": self.target_net.reward,
                "log_prob": self.target_net.running_log_prob,
                "not_terminal": self.target_net.not_terminal,
            }

        def sample_action(self, sess, state):
            result = super().sample_action(sess, state)
            if self.target_net.trainable:
                self.exp["log_prob"].append(result[1])
            return result

        def interact_with_env(self, sess, state, action):
            state_, reward, terminal, info = super().interact_with_env(sess, state, action)
            self.exp["not_terminal"].append(0.0 if terminal and not self.overtime(info) else 1.0)
            return state_, reward, terminal, info

        # def need_train(self, sess, terminal, last_state, info):
        #     self._episodic_training, self.episodic_training = self.episodic_training, self._episodic_training
        #     r = super().need_train(sess, terminal, last_state, info)
        #     self._episodic_training, self.episodic_training = self.episodic_training, self._episodic_training
        #     return r

    return VTraceWorker


SyncVTraceWorker = vtrace_worker_wrapper(SyncActorCriticWorker)
AsyncVTraceWorker = vtrace_worker_wrapper(AsyncActorCriticWorker)
IMPALAWorker = VTraceLearner = vtrace_worker_wrapper(BaseActorCriticLearner)
