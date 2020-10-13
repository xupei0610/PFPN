from .base_worker import SyncActorCriticWorker
from .base_worker import AsyncActorCriticWorker
from .base_worker import BaseActorCriticLearner

__all__ = [
    "SyncA2CWorker", "BatchedA2CWorker",
    "AsyncA2CWorker", "A3CWorker",
    "A2CLearner"
]

def a2c_worker_wrapper(base_actor_critic_worker):
    class A2CWorker(base_actor_critic_worker):

        def __init__(self, **kwargs): 
            kwargs["_on_policy"] = True
            super().__init__(**kwargs)

        @property
        def exp_buffers(self):
            return ["state", "action", "reward", "value", "value_target", "advantage"]
            
        @property
        def train_args(self):
            args = [
                self.exp["state"], self.exp["action"],
                self.exp["value_target"], self.exp["advantage"]
            ]
            if self.sequence_length is not None:
                args.append(self.sequence_length)
            return args

        @property
        def exp_tensors(self):
            return {
                "state": self.target_net.state,
                "action": self.target_net.action,
                "value_target": self.target_net.value_target,
                "advantage": self.target_net.advantage
            }

        def sample_action(self, sess, state):
            result = super().sample_action(sess, state)
            if self.target_net.trainable:
                self.exp["value"].append(result[1])
            return result
        
        def interact_with_env(self, sess, state, action):
            state_, reward, terminal, info = super().interact_with_env(sess, state, action)
            self.last_state = state_
            return state_, reward, terminal, info

        def need_train(self, sess, terminal, last_state, info):
            need_train = super().need_train(sess, terminal, last_state, info)
            if terminal or need_train or self.episodic_terminal:
                if hasattr(self, "sequence_length") and self.sequence_length is not None:
                    if len(self.sequence_length) > 0:
                        self.buffer_stamp = len(self.sequence_length) * max(self.sequence_length)
                    else:
                        self.buffer_stamp = 0
                # td_target as value_target, td_error as advantage
                # td_target as value_target, gae as advantage
                # value+advantage as value_target, gae as advantage
                # reward as value_target, reward-value as advantage      
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
                self.exp["value_target"].extend(self.local_net.value_target_estimate(
                    self.exp["value"][self.buffer_stamp:],
                    self.exp["advantage"][self.buffer_stamp:]
                ))
                self.buffer_stamp = len(self.exp["value"])
            return need_train

        def setup_buffers(self):
            self.buffer_stamp = 0
            return super().setup_buffers()
        
        def clear_buffers(self):
            super().clear_buffers()
            self.buffer_stamp = 0
    return A2CWorker


SyncA2CWorker = BatchedA2CWorker = a2c_worker_wrapper(SyncActorCriticWorker)
AsyncA2CWorker = A3CWorker = a2c_worker_wrapper(AsyncActorCriticWorker)
A2CLearner = a2c_worker_wrapper(BaseActorCriticLearner)
