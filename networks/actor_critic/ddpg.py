import numpy as np
import tensorflow as tf

from .a2c import ContinuousA2CNetwork
from .a2c import ParticleFilteringA2CNetwork

__all__ = [
    "ContinuousDDPGNetwork", "ParticleFilteringDDPGNetwork"
]

def ddpg_network_wrapper(base_class):
    class DDPGNetwork(base_class):
        def __init__(self,
            # False for DDPG
            twin_q=True, 
            # 1 (no policy_delay) for DDPG
            policy_delay=2, tau=0.005,
            # Ornstein Uhlenbeck Action Noise, original DDPG
            ou_noise=False, ou_sigma=0.3, ou_theta=0.15, ou_dt=0.01,
            # Gaussian Noise, TD3, None for learnable
            act_noise=0.1,
            # target policy smoothing, 0 for DDPG, None for learnable
            target_noise=0.2, #noise_clip=0.5,
            **kwargs):
            if "normalize_policy_output" not in kwargs:
                kwargs["normalize_policy_output"] = True

            if kwargs["trainable"]:
                log = kwargs["log"] if "log" in kwargs else None
                kwargs["trainable"] = False
                kwargs["log"] = False
                self.target_net = self.__class__(twin_q=twin_q, act_noise=target_noise, **kwargs)
                self.target_net.setup_value_target_tensor = lambda *args, **kwargs: None
                kwargs["trainable"] = True
                if log is None:
                    del kwargs["log"]
                else:
                    kwargs["log"] = log

            self.ou_noise = ou_noise
            if self.ou_noise:
                self.ou_sigma = ou_sigma
                self.ou_theta = ou_theta
                self.ou_dt = ou_dt
            else:
                self.act_noise = act_noise

            if self.ou_noise or self.act_noise is not None:
                kwargs["fixed_sigma"] = True
                kwargs["init_sigma"] = self.act_noise

            super().__init__(**kwargs)

            self.twin_q = twin_q
            self.policy_delay = policy_delay
            self.tau = tau
            # self.act_noise = act_noise
            self.target_noise = target_noise
            # self.noise_clip = noise_clip
            

        def init(self):
            if hasattr(self, "target_net") and self.policy_delay > 1:
                with tf.variable_scope("target_net_sync_switch"):
                    self.sync_target_net = tf.equal(tf.train.get_global_step() % int(self.policy_delay), 0)

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

                for v, v_ in zip(params, params_):
                    print(v.name, v_.name)
                assert(len(params) == len(params_))

                with tf.variable_scope("sync_"):
                    def sync_target_net_op():
                        return tf.group([
                            tf.assign(v_, (1-self.tau)*v_ + self.tau*v)
                            for v, v_ in zip(params, params_)
                        ])
                    if self.policy_delay > 1:
                        self.train_ops.append(lambda : \
                            tf.cond(self.sync_target_net,
                                sync_target_net_op,
                                lambda : tf.no_op()
                            )                
                        )
                    else:
                        self.train_ops.append(sync_target_net_op)

                with tf.variable_scope("init_"):
                    target_net_init_flag = tf.Variable(False, dtype=tf.bool, trainable=False, name="_init_flag")
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
                if self.policy_delay > 1: self.local_step = 0

                
            
        def setup_advantage_tensor(self):
            return None
            
        # def build_action_log_prob(self, policy, action):
        #     raise NotImplementedError
        
        def build_action_sampler(self, random_action, policy):
            deterministic = not random_action or self.ou_noise or self.act_noise is not None
            self.action_optimal = a = super().build_action_sampler(not deterministic, policy)
            if random_action and deterministic:
                if self.ou_noise:
                    with tf.variable_scope("ou_noise"):
                        noise = tf.get_variable("noise",
                            shape=[1]+list(policy.shape[1:]), dtype=tf.float32,
                            trainable=False, initializer=tf.zeros_initializer()
                        )
                        noise = tf.assign(noise, 
                            (1-self.theta*self.dt)*noise + (self.sigma*np.sqrt(self.dt))*tf.random.normal(tf.shape(noise))
                        )
                    a += noise
                elif (hasattr(self.act_noise, "__len__") and sum(self.act_noise) != 0) or self.act_noise != 0:
                    a += self.act_noise * tf.random.normal(tf.shape(a))

            lb, ub = None, None
            if self.action_lower_bound is not None:
                lb = tf.constant(self.action_lower_bound, dtype=tf.float32, name="action_lower_bound")
            if self.action_upper_bound is not None:
                ub = tf.constant(self.action_upper_bound, dtype=tf.float32, name="action_upper_bound")
            if lb is None and ub is None:
                return a
            if lb is None:
                return tf.minimum(ub, a)
            if ub is None:
                return tf.maximum(lb, a)
            return tf.clip_by_value(a, lb, ub)
            
            
        def build_value(self, trainable, last_layer, critic_net_shape):
            if last_layer.shape.rank != 2:
                last_layer = tf.reshape(last_layer, [-1, last_layer.shape[1:].num_elements()])

            if hasattr(self, "target_net"): # eval net
                a = self.action_optimal # without noise
            else:
                a = self.action # with target noise

            in_layer = tf.concat([last_layer, a], axis=1)
            with tf.variable_scope("q"):
                q_pi = super().build_value(trainable, in_layer, critic_net_shape)

            if hasattr(self, "target_net"): # eval net
                a = self.action_hist
                in_layer = tf.concat([last_layer, a], axis=1)
                with tf.variable_scope("q", reuse=trainable):
                    q_a = super().build_value(trainable, in_layer, critic_net_shape)
                if self.twin_q:
                    with tf.variable_scope("twin_q"):
                        q_a2 = super().build_value(trainable, in_layer, critic_net_shape)
                    return tf.concat([q_a, q_a2, q_pi], axis=1)
                else:
                    return tf.concat([q_a, q_pi], axis=1)
            else: # target net
                if self.twin_q:
                    with tf.variable_scope("twin_q"):
                        q_pi2 = super().build_value(trainable, in_layer, critic_net_shape)
                    q_pi = tf.squeeze(tf.minimum(q_pi, q_pi2), axis=1)
                return q_pi

        def build_policy_loss(self, policy, *_, **__):
            l = -tf.reduce_mean(self.value[:,-1]) # need to skip the critic (q) network parameters during optimization
            if self.policy_delay > 1:
                return tf.cond(self.sync_target_net,
                    lambda : l,
                    lambda : tf.stop_gradient(l)
                )
            return l

        def setup_value_target_tensor(self):
            with tf.variable_scope('target_net'):
                # In DDPG/TD3, the target net has both of the actor and critic nets, while sac only has the critic net
                # if self.normalize_state:
                #     self.target_net.init_state_normalizer = lambda : (self.state_mean, self.state_std, None)
                # if self.normalize_value:
                #     def foo():
                #         self.target_net.normalized_value=self.target_net.normalized_value*self.value_scale + self.value_offset
                #         return self.value_mean, self.value_std, None
                #     self.target_net.value_normalizer = foo
                # self.target_net.build_common_net_op = lambda : self.common_net
                # self.target_net.build_actor_net_op = lambda : self.actor_net
                self.target_net.random_action = False if not self.target_noise and self.target_noise is not None else self.random_action
                self.target_net.init()
            self.reward = tf.placeholder(tf.float32, [None], name="reward")
            self.not_terminal = tf.placeholder(tf.float32, [None], name="not_terminal")
            with tf.name_scope("q_target"):
                q_target = self.reward + self.gamma * self.not_terminal * self.target_net.value
            return q_target

        def build_value_loss(self, q, q_target):
            q_a = q[:,:-1]
            q_target = tf.expand_dims(q_target, axis=-1)
            return super().build_value_loss(q_a, q_target)

        def run(self, sess, state, ops=None):
            running_ops = [self.action]
            feed_dict = {self.state: [state]}
            if ops is not None: running_ops.extend(ops)
            return self._run(sess, running_ops, feed_dict)

        def train(self, sess, optimizer, ops,
                state, action, reward, not_terminal, state_):
            feed_dict = {
                self.state: state,
                self.action_hist: action,
                self.reward: reward,
                self.target_net.state: state_,
                self.not_terminal: not_terminal
            }
            l, r = self._train(sess, optimizer, ops, feed_dict)
            return l, r
    return DDPGNetwork

ContinuousDDPGNetwork = ddpg_network_wrapper(ContinuousA2CNetwork)
ParticleFilteringDDPGNetwork = ddpg_network_wrapper(ParticleFilteringA2CNetwork)

