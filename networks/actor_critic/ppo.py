import tensorflow as tf
import numpy as np

from .a2c import ContinuousA2CNetwork, DiscreteA2CNetwork
from .a2c import ParticleFilteringA2CNetwork

__all__ = [
    "ContinuousClipPPONetwork",
    "DiscreteClipPPONetwork",
    "ParticleFilteringClipPPONetwork"
]


def clip_ppo_network_wrapper(a2c_base_class):
    class ClipPPONetwork(a2c_base_class):
        def __init__(self, epsilon=0.2, **kwargs):
            super().__init__(**kwargs)
            self.epsilon = epsilon
            self.running_log_prob = None
            self.running_value = None
        
        def init(self):
            self.running_value = tf.placeholder(tf.float32, [None], name="value_running")
            self.running_log_prob = tf.placeholder(tf.float32, shape=[None], name="pi_running")
            super().init()

        def build_action_sampler(self, random_action, policy):
            a, self.action_log_prob = super().build_action_sampler_with_log_prob(random_action, policy)
            return a

        def setup_value_target_tensor(self):
            with tf.name_scope("value_target"):
                v = self.advantage + self.running_value
            return v

        def setup_advantage_tensor(self):
            return tf.placeholder(tf.float32, [None], name="advantage")

        def build_value_loss(self, value, value_target):
            with tf.name_scope("vf"):
                vf = tf.square(value-value_target)
            return tf.reduce_mean(self.valid_data_mask(vf))
        
        def build_policy_loss(self, policy, action_hist, advantage):
            with tf.name_scope("pi_target"):
                target_log_prob = self.build_action_log_prob(policy, action_hist)
            with tf.name_scope("ratio"):
                ratio = tf.exp(target_log_prob - self.running_log_prob)
            with tf.name_scope("surrogate"):
                surrogate = ratio * advantage
            with tf.name_scope("clipped_surrogate"):
                clipped_ratio = tf.clip_by_value(ratio, 1.0-self.epsilon, 1.0+self.epsilon)
                clipped_surrogate = clipped_ratio * advantage
            return -tf.reduce_mean(self.valid_data_mask(tf.minimum(surrogate, clipped_surrogate)))
        
        def run(self, sess, state, ops=None):
            feed_dict = {self.state: [state]}
            running_ops = [self.action]
            if self.trainable:
                running_ops.extend([self.action_log_prob, self.value])
            if ops is not None: running_ops.extend(ops)
            return self._run(sess, running_ops, feed_dict)

        def train(self, sess, optimizer, ops, state, action, value, log_prob, advantage):
            feed_dict = {
                self.state: state,
                self.action_hist: action,
                self.running_value: value,
                self.running_log_prob: log_prob,
                self.advantage: advantage
            }
            return self._train(sess, optimizer, ops, feed_dict)

    return ClipPPONetwork


ContinuousClipPPONetwork = clip_ppo_network_wrapper(ContinuousA2CNetwork)
DiscreteClipPPONetwork = clip_ppo_network_wrapper(DiscreteA2CNetwork)
ParticleFilteringClipPPONetwork = clip_ppo_network_wrapper(ParticleFilteringA2CNetwork)
