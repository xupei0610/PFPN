import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from .a2c import ContinuousA2CNetwork, ParticleFilteringA2CNetwork

__all__ = [
    "ContinuousSACNetwork",
    "ParticleFilteringSACNetwork"
]

def sac_network_wrapper(base_class):
    class AbstractSACNetwork(base_class):
        def __init__(self, alpha=0.2, tau=0.005, **kwargs):
            if "normalize_policy_output" not in kwargs:
                kwargs["normalize_policy_output"] = True

            if kwargs["trainable"]:
                log = kwargs["log"] if "log" in kwargs else None
                kwargs["trainable"] = False
                kwargs["log"] = False
                self.target_net = self.__class__(**kwargs)
                self.target_net.setup_value_target_tensor = lambda *args, **kwargs: None
                kwargs["trainable"] = True
                if log is None:
                    del kwargs["log"]
                else:
                    kwargs["log"] = log

            super().__init__(**kwargs)
            self.alpha = alpha
            self.tau = tau

        
        def init(self):
            if hasattr(self, "target_net"):
                with tf.variable_scope("alpha"):
                    self.log_alpha = tf.Variable(0.0, dtype=tf.float32, trainable=True, name="log_alpha")
                    self.alpha = tf.stop_gradient(tf.exp(self.log_alpha))
                    self.add_summary("model/alpha", self.alpha)

            super().init()
                
            if self.trainable:
                parent_scope = tf.get_variable_scope().name
                if len(parent_scope) > 0: parent_scope += "/"
                params_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=parent_scope+"target_net")
                params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=parent_scope)
                params = [p for p in params if p not in params_]
                if len(params) != len(params_):
                    i = 0
                    while i < len(params):
                        if i < len(params_):
                            n = params[i].name[len(parent_scope):]
                            n_ = params_[i].name[len(parent_scope+"target_net/"):]
                            if n == n_:
                                i += 1
                            else:
                                del params[i]
                        else:
                            del params[i]

                # for v, v_ in zip(params, params_):
                #     print(v.name, v_.name)
                assert(len(params) == len(params_))

                with tf.variable_scope("sync_target_net"):
                    def sync_target_net_op():
                        return tf.group([
                            tf.assign(v_, (1-self.tau)*v_ + self.tau*v)
                            for v, v_ in zip(params, params_)
                        ])
                    self.train_ops.append(sync_target_net_op)

                with tf.variable_scope("init_target_net"):
                    target_net_init_flag = tf.Variable(False, dtype=tf.bool, trainable=False, name="target_net_init_flag")
                    def target_net_sync():
                        return tf.group(
                            *[tf.assign(v_, v) for v, v_ in zip(params, params_)],
                            tf.assign(target_net_init_flag, True)
                        )
                    self.init_ops.append(tf.cond(
                        tf.equal(target_net_init_flag, True),
                        lambda: tf.no_op(),
                        target_net_sync
                    ))

        def setup_advantage_tensor(self):
            return None
            
        def run(self, sess, state, ops=None):
            feed_dict = {self.state: [state]}
            running_ops = [self.action]
            if ops is not None: running_ops.extend(ops)
            return self._run(sess, running_ops, feed_dict)

        def train(self, sess, optimizer, ops, state, action, reward, not_terminal, state_):
            feed_dict = {
                self.state: state,
                self.target_net.state: state_,
                self.action_hist: action,
                self.reward: reward,
                self.not_terminal: not_terminal,
            }
            return self._train(sess, optimizer, ops, feed_dict)

        def build_q(self, trainable, last_layer, critic_net_shape):
            a = self.action
            in_layer = tf.concat([last_layer, a], axis=-1)
            with tf.variable_scope("q1"):
                self.q1_a_target = tf.squeeze(super().build_value(trainable, in_layer, critic_net_shape), axis=1)
            with tf.variable_scope("q2"):
                self.q2_a_target = tf.squeeze(super().build_value(trainable, in_layer, critic_net_shape), axis=1)

            if self.trainable:
                a = self.action_hist
                in_layer = tf.concat([last_layer, a], axis=-1)
                with tf.variable_scope("q1", reuse=True):
                    self.q1_a_running = tf.squeeze(super().build_value(trainable, in_layer, critic_net_shape), axis=1)
                with tf.variable_scope("q2", reuse=True):
                    self.q2_a_running = tf.squeeze(super().build_value(trainable, in_layer, critic_net_shape), axis=1)

            with tf.name_scope("v_target"):
                self.vf_target = tf.minimum(self.q1_a_target, self.q2_a_target) \
                    - self.alpha*self.target_log_prob
                self.vf_target = tf.stop_gradient(self.vf_target)

        def build_action_sampler(self, random_action, policy):
            a, self.target_log_prob = super().build_action_sampler_with_log_prob(random_action, policy)
            return a

        def setup_value_target_tensor(self):
            with tf.variable_scope('target_net'):
                self.setup_target_net()
            self.reward = tf.placeholder(tf.float32, [None], name="reward")
            self.not_terminal = tf.placeholder(tf.float32, [None], name="not_terminal")
            with tf.name_scope("q_target"):
                q_target = self.reward + self.gamma * self.not_terminal * self.target_net.value
            return q_target # unnormalized

        def setup_target_net(self):
            if self.normalize_state:
                self.target_net.init_state_normalizer = lambda : (self.state_mean, self.state_std, None)
            if self.normalize_value:
                def foo():
                    self.target_net.normalized_value=self.target_net.normalized_value*self.value_scale + self.value_offset
                    return self.value_mean, self.value_std, None
                self.target_net.value_normalizer = foo
            self.target_net.alpha = self.alpha
            self.target_net.build_common_net_op = lambda : self.common_net
            self.target_net.build_actor_net_op = lambda : self.actor_net
            self.target_net.random_action = self.random_action
            self.target_net.init()

        def build_value(self, trainable, last_layer, critic_net_shape):
            self.build_q(trainable, last_layer, critic_net_shape)
            if hasattr(self, "target_net"): return None
            return self.vf_target

        def build_value_loss(self, _, q_target):
            q1_loss = tf.square(q_target - self.q1_a_running)
            q2_loss = tf.square(q_target - self.q2_a_running)
            l = q1_loss + q2_loss
            return tf.reduce_mean(self.valid_data_mask(l))

        def build_policy_loss(self, policy, action_hist, _):
            # need to skip the critic (q) network parameters during optimization
            l = self.alpha * self.target_log_prob - tf.minimum(self.q1_a_target, self.q2_a_target)
            # update alpha
            target_entropy = -np.prod([int(_) for _ in action_hist.shape[1:]])
            l -= self.log_alpha * tf.stop_gradient(self.target_log_prob+target_entropy)
            l = tf.reduce_mean(self.valid_data_mask(l))
            return l

    return AbstractSACNetwork



ContinuousSACNetwork = sac_network_wrapper(ContinuousA2CNetwork)
ParticleFilteringSACNetwork = sac_network_wrapper(ParticleFilteringA2CNetwork)
