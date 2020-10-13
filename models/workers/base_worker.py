import time
import tensorflow as tf
import numpy as np

from ..sync_model import SyncWorker
from ..async_model import AsyncWorker
from ..learner_model import AbstractLearner


def actor_critic_worker_wrapper(base_distributed_worker_class):
    class ActorCriticWorker(base_distributed_worker_class):

        def __init__(self, lr_critic, lr_actor, separate_optimizer=False, norm_clip=None, **kwargs):
            super().__init__(**kwargs)

            self.lr_critic = lr_critic
            self.lr_actor = lr_actor

            self.separate_optimizer = separate_optimizer
            self.norm_clip = norm_clip

            self.log_interval_time = 10.0
            self.last_notify_time = time.clock() - self.log_interval_time
        
        def build_optimizer(self, local_net, local_vars, target_vars):
            if callable(self.lr_actor):
                if self.lr_actor == self.lr_critic:
                    self.lr_critic = self.lr_actor = self.lr_actor()
                else:
                    self.lr_actor = self.lr_actor()
            if callable(self.lr_critic): self.lr_critic = self.lr_critic()
            class Opt(tf.train.Optimizer):
                def __init__(self, c_optimizer, a_optimizer):
                    self.c_optimizer = c_optimizer
                    self.a_optimizer = a_optimizer
                def apply_gradients(self, grads_and_vars, global_step=None, name=None):
                    grads_and_vars = list(grads_and_vars)
                    half = int(len(grads_and_vars)/2)
                    with tf.control_dependencies([
                        self.c_optimizer.apply_gradients(grads_and_vars[:half])
                    ]):
                        opt = self.a_optimizer.apply_gradients(grads_and_vars[half:], global_step=global_step)
                    return opt
            with tf.variable_scope("optimizer"):
                with tf.variable_scope("grads"):
                    if self.separate_optimizer or self.lr_critic != self.lr_actor or hasattr(self.norm_clip, "__len__"):
                        value_loss = local_net.value_loss 
                        policy_loss = local_net.policy_loss
                        critic_vars, actor_vars = [], []
                        for v in local_vars:
                            if "critic" not in v.name.split("/"):
                                actor_vars.append(v)
                            if "actor" not in v.name.split("/"):
                                critic_vars.append(v)
                        c_grads = tf.gradients(value_loss, critic_vars)
                        a_grads = tf.gradients(policy_loss, actor_vars)
                        for i, v in enumerate(local_vars):
                            if v not in actor_vars:
                                a_grads.insert(i, None)
                            if v not in critic_vars:
                                c_grads.insert(i, None)
                        assert(len(c_grads) == len(a_grads))
                        grads_and_vars = list(zip(c_grads, target_vars)) + list(zip(a_grads, target_vars))
                    else:
                        loss = local_net.loss
                        grads = tf.gradients(loss, local_vars)
                        grads_and_vars = list(zip(grads, target_vars))
                if self.lr_actor == self.lr_critic and not self.separate_optimizer:
                    lr = tf.constant(self.lr_critic, dtype=tf.float32, name="lr") if isinstance(self.lr_critic, float) else self.lr_critic
                    optimizer = tf.train.AdamOptimizer(lr)
                else:
                    lr_critic = tf.constant(self.lr_critic, dtype=tf.float32, name="lr_critic") if isinstance(self.lr_critic, float) else self.lr_critic
                    lr_actor = tf.constant(self.lr_actor, dtype=tf.float32, name="lr_actor") if isinstance(self.lr_actor, float) else self.lr_actor
                    c_optimizer = tf.train.AdamOptimizer(lr_critic)                            
                    a_optimizer = tf.train.AdamOptimizer(lr_actor)
                    optimizer = Opt(c_optimizer, a_optimizer)
            return optimizer, grads_and_vars

        def clip_grads(self, grads_and_vars):
            grads, target_vars = list(zip(*grads_and_vars)) 
            if self.lr_critic != self.lr_actor or hasattr(self.norm_clip, "__len__"):
                assert(len(grads)/2 == int(len(grads)/2))
                c_grads = grads[:int(len(grads)/2)]
                a_grads = grads[int(len(grads)/2):]
            if hasattr(self.norm_clip, "__len__"):
                if "critic" in self.norm_clip and self.norm_clip["critic"]:
                    c_clipped_grads, self.global_norm_critic = tf.clip_by_global_norm(c_grads, self.norm_clip["critic"])
                else:
                    self.global_norm_critic = tf.global_norm(c_grads)
                    c_clipped_grads = c_grads
                if "actor" in self.norm_clip and self.norm_clip["actor"]:
                    a_clipped_grads, self.global_norm_actor = tf.clip_by_global_norm(a_grads, self.norm_clip["actor"])
                else:
                    self.global_norm_actor = tf.global_norm(a_grads)
                    a_clipped_grads = a_grads
                grads_and_vars = list(zip(c_clipped_grads, target_vars)) + list(zip(a_clipped_grads, target_vars))
            elif self.lr_critic == self.lr_actor:
                if self.norm_clip:
                    clipped_grads, global_norm = tf.clip_by_global_norm(grads, self.norm_clip)
                    grads_and_vars = list(zip(clipped_grads, target_vars))
                    self.global_norm_actor = global_norm
                    self.global_norm_critic = global_norm
                else:
                    grads_and_vars = list(zip(grads, target_vars))
                    global_norm = tf.global_norm(grads)
                    self.global_norm_actor = global_norm
                    self.global_norm_critic = global_norm
            else:
                if self.norm_clip:
                    clipped_grads, self.global_n = tf.clip_by_global_norm(c_grads+a_grads, self.norm_clip)
                    c_clipped_grads = clipped_grads[:len(c_grads)]
                    a_clipped_grads = clipped_grads[len(c_grads):]
                    self.global_norm_actor = tf.global_norm(a_grads)
                    self.global_norm_critic = tf.global_norm(c_grads)
                else:
                    c_clipped_grads, a_clipped_grads = c_grads, a_grads
                    self.global_norm_actor = tf.global_norm(a_grads)
                    self.global_norm_critic = tf.global_norm(c_grads)
                grads_and_vars = list(zip(c_clipped_grads, target_vars)) + list(zip(a_clipped_grads, target_vars))
            return grads_and_vars

        def build_summaries(self, target_net):
            s = super().build_summaries(target_net)
            # if self.global_norm_actor == self.global_norm_critic:
            #     s.append(tf.summary.scalar("model/global_norm", self.global_norm_actor))
            # else:
            #     s.append(tf.summary.scalar("model/global_norm_actor", self.global_norm_actor))
            #     s.append(tf.summary.scalar("model/global_norm_critic", self.global_norm_critic))
            # if self.lr_critic == self.lr_actor:
            #     s.append(tf.summary.scalar("model/lr", self.lr_actor))
            # else:
            #     s.append(tf.summary.scalar("model/lr_actor", self.lr_actor))
            #     s.append(tf.summary.scalar("model/lr_critic", self.lr_critic))
            return s
            
        def train(self, *args, **kwargs):
            loss, result = super().train(*args, **kwargs)
            if self.logger is not None and loss is not None:
                _, policy_entropy, policy_loss, value_loss = loss
                current_clock = time.clock()
                if np.isnan(policy_loss) or np.isnan(value_loss) or current_clock > self.log_interval_time + self.last_notify_time:
                   ops = self.train_ops
                   print("[TRAIN] Step: {}; Value Loss: {:.4f}; Policy Loss: {:.4f}; Policy Entropy: {:.4f}; "
                       "Worker: {}; {}".format(
                           result[ops.index(self.global_step)],
                           value_loss, policy_loss,
                           0.0 if policy_entropy is None else policy_entropy,
                           self.name, time.strftime("%m-%d %H:%M:%S", time.localtime())))
                   self.last_notify_time = current_clock
            return loss, result
        
        @staticmethod
        def overtime(info):
            return "TimeLimit.truncated" in info and info["TimeLimit.truncated"]

    return ActorCriticWorker


SyncActorCriticWorker = actor_critic_worker_wrapper(SyncWorker)
AsyncActorCriticWorker = actor_critic_worker_wrapper(AsyncWorker)
BaseActorCriticLearner = actor_critic_worker_wrapper(AbstractLearner)
