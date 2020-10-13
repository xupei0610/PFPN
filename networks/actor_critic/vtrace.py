import numpy as np
import tensorflow as tf

from .a2c import ContinuousA2CNetwork, DiscreteA2CNetwork
from .a2c import ParticleFilteringA2CNetwork

__all__ = [
    "ContinuousVTraceNetwork",
    "DiscreteVTraceNetwork",
    "ParticleFilteringVTraceNetwork"
]


def v_trace_network_wrapper(base_a2c_class):
    class VTraceNetwork(base_a2c_class):
        def __init__(self, rho_clip=1.0, pg_rho_clip=1.0, **kwargs):
            # all data fed into the network is assumed in order of (B*T, x)
            #
            # batch_size represents the number of episodes of the input data
            # it has the same concept with that in LSTM or RNN
            #
            # V-trace computes value_target using the target network (online)
            # it needs to know the last value for each episode
            # whose data are parallelly fed into the network with or without stacking
            #
            # Similarly to LSTM or RNN,
            # it is required the input data from each episode have the same length
            # and thus zero-padding is needed for those with insufficient length
            #
            # In the default behavior of sync and async model,
            # data are always fed one episode by one episode
            # and, thereby, batch_size = 1
            # In the default behavior of learner-centered model,
            # multiple actors collect data simultaneously,
            # the learner many perform training using data from multiple actors at once,
            # and, thereby, batch_size = #{actors from which training data are collected at once}
            #
        
            super().__init__(**kwargs)

            self.rho_clip = rho_clip
            self.pg_rho_clip = pg_rho_clip
            self.episodic_batch_size = None
            self.batch_sequence_length = None
            
        def init(self):
            if self.trainable:
                self.reward = tf.placeholder(tf.float32, [None], name="reward")
                self.running_log_prob = tf.placeholder(tf.float32, shape=[None], name="pi_running")
                self.not_terminal = tf.placeholder(tf.float32, [None], name="not_terminal")
            if self.episodic_batch_size is None and self.batch_sequence_length is None:
                self.batch_sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
            assert(self.episodic_batch_size is not None or self.batch_sequence_length is not None)
            super().init()


        def to_batches(self, tensor):
            if self.episodic_batch_size is None: self.setup_batch_size_tensor(self.state)
            # reshape the given tensor to [B, N, ...] from [BxN, ...] and
            # transpose it to [N, B, ...] shape
            rs = tf.reshape(tensor, tf.concat(([self.episodic_batch_size, -1], tf.shape(tensor)[1:]), axis=0))
            return tf.transpose(rs, [1, 0] + list(range(rs.shape.ndims))[2:])
        
        def to_flat(self, tensor):
            return tf.reshape(tf.transpose(tensor, [1, 0] + list(range(tensor.shape.ndims))[2:]), tf.concat(([-1], tf.shape(tensor)[2:]), axis=0))

        def setup_batch_size_tensor(self, state):
            with tf.variable_scope("batch_size"):
                if self.trainable and self.batch_sequence_length is not None:
                    if self.episodic_batch_size is None:
                        with tf.name_scope("time_major_batch_size"):
                            self.episodic_batch_size = tf.shape(self.batch_sequence_length)[0]
                    with tf.name_scope("valid_batch_mask"):
                        rollout_len = tf.cast(tf.shape(state)[0] / self.episodic_batch_size, tf.int32)
                        self.episodic_end_idx = rollout_len * tf.range(self.episodic_batch_size) + (self.batch_sequence_length-1)
                        with tf.name_scope("batched_episodic_end_mask"):
                            self.batched_episodic_end_mask = tf.stop_gradient(tf.sparse_to_dense(
                                tf.stack([self.batch_sequence_length-1, tf.range(self.episodic_batch_size)], axis=1),
                                [rollout_len, self.episodic_batch_size],
                                True, False, validate_indices=False))
                        batch_sequence_length = self.batch_sequence_length
                        self.mask = tf.stop_gradient(tf.reshape(tf.sequence_mask(batch_sequence_length, rollout_len), [-1]))
                    self.valid_data_mask = lambda x: tf.boolean_mask(x, self.mask)
                else:
                    self.episodic_end_idx = None

        def build_action_sampler(self, random_action, policy):
            action = super().build_action_sampler(random_action, policy)
            with tf.name_scope("action_logp"):
                self.action_log_prob = self.build_action_log_prob(policy, action)
            return action
        
        def concat_bootstrap_value(self, v, value):
            v_t1 = tf.concat((v[1:], tf.expand_dims(value[-1], 0)), axis=0)
            if self.episodic_end_idx is not None:
                v_t1 = tf.where(self.batched_episodic_end_mask,
                    value, v_t1
                )
            return v_t1
        
        def setup_vtrace(self):
            with tf.name_scope("pi_target"):
                self.target_log_prob = self.build_action_log_prob(self.policy, self.action_hist)
                
            with tf.name_scope("vtrace"):
                with tf.name_scope("batched_target_log_prob"):
                    target_log_prob = self.to_batches(self.target_log_prob)
                with tf.name_scope("batched_behavior_log_prob"):
                    running_log_prob = self.to_batches(self.running_log_prob)
                with tf.name_scope("batched_value"):
                    self.batched_value = self.to_batches(self.value)
                with tf.name_scope("batched_reward"):
                    self.batched_reward = self.to_batches(self.reward)
                with tf.name_scope("batched_gamma"):
                    gamma = tf.constant(self.gamma, dtype=tf.float32, name="gamma_{}".format(self.gamma))
                    self.batched_discount = gamma * self.to_batches(self.not_terminal)
        
                with tf.name_scope("rho"):
                    log_diff = target_log_prob - running_log_prob
                    self.rho = tf.exp(log_diff)

                with tf.name_scope("rho"):
                    if self.rho_clip is None:
                        v_rho = self.rho
                    else:
                        with tf.name_scope("clip"):
                            rho_clip = tf.constant(self.rho_clip, dtype=tf.float32, name="rho_clip_{}".format(self.rho_clip))
                            v_rho = tf.minimum(rho_clip, self.rho)
                with tf.name_scope("value_t1"):
                    value_t1 = self.concat_bootstrap_value(self.batched_value, self.batched_value)
                with tf.name_scope("td_error"):
                    value_td_err = self.batched_reward + self.batched_discount * value_t1 - self.batched_value
                    with tf.name_scope("scale"):
                        value_td_err = v_rho * value_td_err
                    with tf.name_scope("c"):
                        c = tf.minimum(1.0, self.rho)
                    with tf.name_scope("vs_minus_value"):
                        batched_discount = self.batched_discount
                        if self.episodic_end_idx is not None:
                            batched_discount = tf.where(self.batched_episodic_end_mask,
                                tf.zeros_like(self.batched_discount), batched_discount, 
                            )
                        elems = (batched_discount, c, value_td_err)
                        def discount_fn(a, elem):
                            discount_t, c_t, td_err_t = elem
                            return td_err_t + discount_t * c_t * a
                        vs_minus_value = tf.scan(
                            discount_fn, elems,
                            initializer=tf.zeros_like(value_td_err[0]),
                            parallel_iterations=1, back_prop=False,
                            reverse=True
                        )
                    self.vs = tf.add(vs_minus_value, self.batched_value)


        def setup_value_target_tensor(self):
            if not hasattr(self, "vs"): self.setup_vtrace()
            with tf.name_scope("value_target"):
                vs = self.to_flat(self.vs)
            return vs
        
        def setup_advantage_tensor(self):
            if not hasattr(self, "vs"): self.setup_vtrace()

            with tf.name_scope("advantage"):
                with tf.name_scope("rho"):
                    if self.pg_rho_clip is None:
                        pg_rho = self.rho
                    else:
                        with tf.name_scope("clip"):
                            rho_clip = tf.constant(self.pg_rho_clip, dtype=tf.float32, name="pg_rho_clip_{}".format(self.pg_rho_clip))
                            pg_rho = tf.minimum(rho_clip, self.rho)
                with tf.name_scope("vs_t1"):
                    vs_t1 = self.concat_bootstrap_value(self.vs, self.batched_value)
                with tf.name_scope("td_error"):
                    vs_td_err = self.batched_reward + self.batched_discount * vs_t1 - self.batched_value
                    with tf.name_scope("scale"):
                        vs_td_err = pg_rho * vs_td_err
                pg_adv = self.to_flat(vs_td_err)
            return pg_adv

        def build_policy_loss(self, policy, action_hist, advantage):
            return -tf.reduce_mean(self.valid_data_mask(tf.multiply(self.target_log_prob, advantage)))

        def run(self, sess, state, ops=None):
            feed_dict = {self.state: [state]}
            if self.batch_sequence_length is not None:
                feed_dict[self.batch_sequence_length] = [1]
            running_ops = [self.action]
            if self.trainable:
                running_ops.append(self.action_log_prob)
            if ops is not None: running_ops.extend(ops)
            return self._run(sess, running_ops, feed_dict)

        def train(self, sess, optimizer, ops, state, action, reward, log_prob, not_terminal, batch_sequence_length=None):
            feed_dict = {
                self.state: state,
                self.action_hist: action,
                self.reward: reward,
                self.running_log_prob: log_prob,
                self.not_terminal: not_terminal
            }
            if self.batch_sequence_length is not None:
                feed_dict[self.batch_sequence_length] = batch_sequence_length
            r = self._train(sess, optimizer, ops, feed_dict)
            return r
    return VTraceNetwork

ContinuousVTraceNetwork = v_trace_network_wrapper(ContinuousA2CNetwork)
DiscreteVTraceNetwork = v_trace_network_wrapper(DiscreteA2CNetwork)
ParticleFilteringVTraceNetwork = v_trace_network_wrapper(ParticleFilteringA2CNetwork)
