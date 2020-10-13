from .base_worker import SyncActorCriticWorker
from .base_worker import AsyncActorCriticWorker
from .base_worker import BaseActorCriticLearner

__all__ = [
    "SyncPPOWorker", "DPPOWorker",
    "AsyncPPOWorker",
    "PPOLearner"
]

def ppo_worker_wrapper(base_actor_critic_worker):
    class PPOWorker(base_actor_critic_worker):
        def __init__(self, **kwargs): 
            kwargs["_on_policy"] = True
            super().__init__(**kwargs)

        @property
        def exp_buffers(self):
            return ["state", "action", "reward", "value", #"value_target",
            "advantage", "log_prob"]
            
        @property
        def train_args(self):
            args = [
                self.exp["state"], self.exp["action"], self.exp["value"],
                self.exp["log_prob"], self.exp["advantage"] #self.exp["value_target"]
            ]
            if self.sequence_length is not None:
                args.append(self.sequence_length)
            return args

        @property
        def exp_tensors(self):
            return {
                "state": self.target_net.state, 
                "action": self.target_net.action,
                "value": self.target_net.running_value,
                "log_prob": self.target_net.running_log_prob,
                # "value_target": self.target_net.value_target
                "advantage": self.target_net.advantage
            }

        def sample_action(self, sess, state):
            result = super().sample_action(sess, state)
            if self.target_net.trainable:
                self.exp["log_prob"].append(result[1])
                self.exp["value"].append(result[2])
            return result

        def need_train(self, sess, terminal, last_state, info):
            need_train = super().need_train(sess, terminal, last_state, info)
            if terminal or need_train or self.episodic_terminal:
                if hasattr(self, "sequence_length") and self.sequence_length is not None:
                    if len(self.sequence_length) > 0:
                        self.buffer_stamp = len(self.sequence_length) * max(self.sequence_length)
                    else:
                        self.buffer_stamp = 0
                if terminal and not self.overtime(info):
                    boostrap_value = 0.0
                else:
                    boostrap_value = self.local_net.evaluate(sess, last_state)
                self.exp["value"].append(boostrap_value)
                self.exp["advantage"].extend(self.local_net.generalized_advantage_estimate(
                    self.exp["reward"][self.buffer_stamp:],
                    self.exp["value"][self.buffer_stamp:]
                ))
                self.exp["value"].pop()
                # self.exp["value_target"].extend(self.local_net.value_target_estimate(
                #     self.exp["value"][self.buffer_stamp:],
                #     self.exp["advantage"][self.buffer_stamp:]
                # ))
                self.buffer_stamp = len(self.exp["value"])
            return need_train
        
        def setup_buffers(self):
            self.buffer_stamp = 0
            return super().setup_buffers()
        
        def clear_buffers(self):
            super().clear_buffers()
            self.buffer_stamp = 0
    return PPOWorker


SyncPPOWorker = DPPOWorker = ppo_worker_wrapper(SyncActorCriticWorker)
AsyncPPOWorker = ppo_worker_wrapper(AsyncActorCriticWorker)
PPOLearner = ppo_worker_wrapper(BaseActorCriticLearner)
