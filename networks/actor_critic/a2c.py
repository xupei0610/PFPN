import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os

from .actor_critic import ActorCriticNetwork 
from ..ops import *
from ..utils import discount, MixtureGaussianDistribution

__all__ = [
    "ContinuousA2CNetwork",
    "DiscreteA2CNetwork",
    "ParticleFilteringA2CNetwork"
]

class A2CNetwork(ActorCriticNetwork):

    def __init__(self, gamma=0.99, lambd=0.95, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.gae_gamma = None if lambd is None else self.gamma * lambd
    
    def td_target(self, reward, boostrap_value, normalize=False):
        return discount(reward, self.gamma, boostrap_value, normalize)
    
    def td_error(self, td_target, value, normalize=False):
        td_err = np.subtract(td_target, value)
        return td_err if not normalize else ((td_err - np.mean(td_err)) / (np.std(td_err) + 1e-8))

    def generalized_advantage_estimate(self, reward, value, normalize=False):
        assert(len(value) == len(reward)+1)
        r = np.asarray(reward, dtype=np.float32)
        v = np.asarray(value[:-1], dtype=np.float32)
        v_ = np.asarray(value[1:], dtype=np.float32)
        td_err = r+self.gamma*v_-v

        if self.gae_gamma:
            return discount(td_err, self.gae_gamma, 0, normalize)
        else:
            return td_err if not normalize else ((td_err - np.mean(td_err)) / (np.std(td_err) + 1e-8))
    
    def value_target_estimate(self, value, advantage, normalize=False):
        result = np.add(value, advantage)
        if normalize:
            result = np.subtract(result, np.mean(result))
            result = np.divide(result, np.std(result)+1e-6)
            return result[::-1]
        else:
            return result

    def build_value_loss(self, value, value_target):
        d = tf.square(self.valid_data_mask(value-value_target))
        if d.shape.ndims > 1:
            assert(d.shape.ndims == 2)
            d = tf.reduce_sum(d, axis=1)
        return tf.reduce_mean(d)
    
    def build_policy_loss(self, policy, action_hist, advantage):
        target_log_prob = self.build_action_log_prob(policy, action_hist)
        return -tf.reduce_mean(self.valid_data_mask(tf.multiply(target_log_prob, advantage)))

    def build_policy_entropy(self, policy):
        ent = policy.entropy()
        if len(ent.shape) > 1: ent = tf.reduce_sum(ent, axis=1)
        return tf.reduce_mean(self.valid_data_mask(ent))

    def run(self, sess, state, ops=None):
        feed_dict = {self.state: [state]}
        running_ops = [self.action]
        if self.trainable:
            running_ops.append(self.value)
        if ops is not None: running_ops.extend(ops)
        return self._run(sess, running_ops, feed_dict)

    def evaluate(self, sess, state):
        ops = [self.value]
        feed_dict = {self.state: [state]}
        return self._run(sess, ops, feed_dict)[0]
    
    def train(self, sess, optimizer, ops,
              state, action, value_target, advantage):
        feed_dict = {
            self.state: state,
            self.action_hist: action,
            self.value_target: value_target,
            self.advantage: advantage
        }
        return self._train(sess, optimizer, ops, feed_dict)


class ContinuousA2CNetwork(A2CNetwork):

    def __init__(self, action_lower_bound=None, action_upper_bound=None,
                 init_sigma=None, max_sigma=None, sigma_eps=2e-9, fixed_sigma=False, variabilize_sigma=False,
                 normalize_policy_output=False,
                 **kwargs):
        super().__init__(**kwargs)
        if hasattr(self.action_shape, "__len__"):
            assert(len(self.action_shape) == 1)
            n_actions = self.action_shape[0]
        else:
            n_actions = self.action_shape
        self.mu = None
        if init_sigma is None:
            self.init_sigma = None
        elif hasattr(init_sigma, "__len__"):
            self.init_sigma = init_sigma
        else:
            self.init_sigma = [init_sigma]*n_actions
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound
        self.fixed_sigma = fixed_sigma
        self.variabilize_sigma = variabilize_sigma
        self.max_sigma = max_sigma
        self.sigma_eps = sigma_eps
        self.normalize_policy_output = normalize_policy_output
        
    def init(self):
        if np.isinf(np.sum(self.action_lower_bound)):
            self.action_lower_bound = None
        if np.isinf(np.sum(self.action_upper_bound)):
            self.action_upper_bound = None
        if self.fixed_sigma is False: self.fixed_sigma = None
        if self.fixed_sigma is True: self.fixed_sigma = self.init_sigma
        if self.fixed_sigma is None:
            if self.variabilize_sigma is False: self.variabilize_sigma = None
            if self.variabilize_sigma is True: self.variabilize_sigma = self.init_sigma
            if self.variabilize_sigma is not None:
                self.init_sigma = self.variabilize_sigma
                self.max_sigma = None
        else:
            self.init_sigma = self.fixed_sigma
            self.variabilize_sigma = False
            self.max_sigma = None
            self.fixed_sigma = True
            if self.init_sigma == 0:
                self.random_action = False

        self.normalize_policy_output = self.normalize_policy_output and self.action_lower_bound is not None and self.action_upper_bound is not None 
        
        super().init()
        if self.action is not None:
            for n, a in zip(range(self.action.shape[-1]), tf.split(self.valid_data_mask(self.action), [1]*self.action.shape[-1], axis=-1)):
                self.add_summary("cont_action/{}".format(n), a)

    def build_policy(self, trainable, last_layer, action_shape):
        if hasattr(action_shape, "__len__"):
            assert(len(action_shape) == 1)
            n_actions = action_shape[0]
        else:
            n_actions = action_shape
        index = 2 if last_layer.shape[1].value is None else 1

        self.mu = fc_layer("fc_mu", last_layer,
            last_layer.shape[index:].num_elements(), n_actions,
            weight_initializer=self.weight_initializer, activator=None,
            trainable=trainable
        )
        self.add_summary("action/mu", self.mu, self.valid_data_mask)

        if self.fixed_sigma is not None:
            self.sigma = self.init_sigma
        else:
            if self.variabilize_sigma and not self.init_sigma:
                self.init_sigma = float(self.variabilize_sigma)
            if self.max_sigma:
                activator = tf.tanh
                offset = 0.5*(np.log(self.max_sigma)+np.log(self.sigma_eps))
                scale = 0.5*(np.log(self.max_sigma)-np.log(self.sigma_eps))
                if self.init_sigma:
                    init_log_sigma = (np.log(self.init_sigma)-offset)/scale
                    init_log_sigma = 0.5*(np.log(1+init_log_sigma)-np.log(1-init_log_sigma))
            else:
                activator = None
                if self.init_sigma:
                    init_log_sigma = np.expand_dims(np.log(self.init_sigma), 0)

            if self.variabilize_sigma:
                log_sigma = tf.Variable(init_log_sigma, dtype=tf.float32, name="log_sigma", trainable=trainable)
                if activator: log_sigma = activator(log_sigma)
            elif self.init_sigma:
                log_sigma = fc_layer("fc_log_sigma", last_layer,
                    last_layer.shape[index:].num_elements(), n_actions,
                    weight_initializer=tf.constant_initializer(0.0),
                    bias_initializer=tf.constant_initializer(init_log_sigma),
                    activator=activator,
                    trainable=trainable)
            else:
                log_sigma = fc_layer("fc_log_sigma", last_layer,
                    last_layer.shape[index:].num_elements(), n_actions,
                    weight_initializer=self.weight_initializer, activator=activator,
                    trainable=trainable)

            if self.max_sigma:
                if scale != 1: log_sigma = scale*log_sigma
                if offset != 0: log_sigma = offset+log_sigma
                self.sigma = tf.exp(log_sigma)
            else:
                self.sigma = tf.exp(log_sigma)
                self.sigma += self.sigma_eps
            self.add_summary("action/sigma", self.sigma, self.valid_data_mask)

        return tfp.distributions.Normal(self.mu, self.sigma, allow_nan_stats=False)

    def deterministic_action(self, policy):
        return policy.mean()
    
    def build_action_sampler(self, random_action, policy):
        a = policy.sample(1) if random_action else self.deterministic_action(policy)
        if hasattr(a, "__len__"): a = a[0]
        if random_action: a = tf.squeeze(a, axis=0)
        if self.normalize_policy_output: a = tf.nn.tanh(a)
        return self.denormalize_action(a)
    
    def build_action_log_prob(self, policy, action):
        normalized_action = self.normalize_action(action)
        if self.normalize_policy_output:
            with tf.name_scope("inv_tanh"):
                action = tf.math.atanh(normalized_action)
        logp = super().build_action_log_prob(policy, action)
        if self.normalize_policy_output:
            logp -= tf.reduce_sum(tf.log(clip_by_value_with_gradient(1-normalized_action**2, 0, 1)+1e-6), axis=1)
        return logp

    def build_action_sampler_with_log_prob(self, random_action, policy):
        a = policy.sample(1) if random_action else self.deterministic_action(policy)
        if random_action:
            if hasattr(a, "__len__"):
                a = tuple([tf.squeeze(_ , axis=0) for _ in a])
            else:
                a = tf.squeeze(a , axis=0)
        action = a[0] if hasattr(a, "__len__") else a

        if self.normalize_policy_output:
            normalized_action = tf.nn.tanh(action)
            with tf.name_scope("action_logp"):
                logp = super().build_action_log_prob(policy, a)
                logp -= tf.reduce_sum(2*(np.log(2., dtype=np.float32) - action - tf.nn.softplus(-2*action)), axis=1)
        else:
            normalized_action = action
            with tf.name_scope("action_logp"):
                logp = super().build_action_log_prob(policy, a)
        return self.denormalize_action(normalized_action), logp


    def denormalize_action(self, a):
        if self.normalize_policy_output:
            with tf.name_scope("denormalize_action"):
                gap = np.subtract(self.action_upper_bound, self.action_lower_bound)
                for _ in gap:
                    if _ != 2:
                        scale = tf.constant(np.multiply(gap, 0.5), dtype=tf.float32, name="scale")
                        a = scale * a
                        break
                s = np.add(self.action_upper_bound, self.action_lower_bound)
                for _ in s:
                    if _ != 0:
                        offset = tf.constant(np.multiply(s, 0.5), dtype=tf.float32, name="offset")
                        a = a + offset
                        break
        return a

    def normalize_action(self, action):
        if self.normalize_policy_output:
            with tf.name_scope("normalize_action"):
                s = np.add(self.action_upper_bound, self.action_lower_bound)
                for _ in s:
                    if _ != 0:
                        offset = tf.constant(np.multiply(s, 0.5), dtype=tf.float32, name="offset")
                        action = action - offset
                        break
                gap = np.subtract(self.action_upper_bound, self.action_lower_bound)
                for _ in gap:
                    if _ != 2:
                        inv_scale = tf.constant(1.0/np.multiply(gap, 0.5), dtype=tf.float32, name="scale")
                        action = action * inv_scale
                        break
        return action


class DiscreteA2CNetwork(A2CNetwork):
    def build_policy(self, trainable, last_layer, action_shape):
        if hasattr(action_shape, "__len__"):
            assert(next((False for n in action_shape if n != action_shape[0]), True))
            n_actions = sum(action_shape)
        else:
            n_actions = action_shape
            action_shape = [n_actions]
        index = 2 if last_layer.shape[1].value is None else 1
        policy_logits = fc_layer("fc_policy", last_layer,
                    last_layer.shape[index:].num_elements(), n_actions,
                    weight_initializer=self.weight_initializer, activator=None,
                    trainable=trainable)
        if len(action_shape) > 1:
            policy_logits = tf.reshape(policy_logits, [-1, len(action_shape), action_shape[0]])
        self.action_shape = [len(action_shape)]
        return tf.distributions.Categorical(logits=policy_logits)

    def build_action_sampler(self, random_action, policy):
        a = tf.squeeze(policy.sample(1), axis=0) if random_action else tf.argmax(policy.logits, axis=-1)
        return a

    def build_action_sampler_with_log_prob(self, random_action, policy):
        action = DiscreteA2CNetwork.build_action_sampler(self, random_action, policy)
        with tf.name_scope("action_logp"):
            log_prob = self.build_action_log_prob(policy, action)
        return action, log_prob


class ParticleFilteringA2CNetwork(ContinuousA2CNetwork):
    def __init__(self, action_lower_bound, action_upper_bound, init_sigma=None, fixed_sigma=False,
                 particles=50, resample=3, resample_interval=2000, resample_threshold=None,
                 normalize_policy_output=False, **kwargs):
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound
        A2CNetwork.__init__(self, **kwargs)
        if len(self.action_shape) != 1:
            raise ValueError("Particle Filtering Policy Network only supports continuous action space.")

        self.dis_action_shape = [particles] * self.action_shape[0]
        self.resample = resample
        self.resample_interval = resample_interval
        self.resample_threshold = resample_threshold
        self.init_sigma = fixed_sigma if fixed_sigma and fixed_sigma is not True else init_sigma
        self.fixed_sigma = True if fixed_sigma else False

        self.normalize_policy_output = False # tanh squash after mixture
        self.normalize_policy_output_ = normalize_policy_output # tanh squash before mixture
        
    def init(self):
        self.identical_n_particles = next((False for n in self.dis_action_shape if n != self.dis_action_shape[0]), True)
        if not self.identical_n_particles:
            raise NotImplementedError("Current Implementation only suppots each action dimension havign the same number of particles.")
        A2CNetwork.init(self)

        if self.trainable:
            for n, p, pstd, ca in zip(range(len(self.dis_action_shape)),
                [tf.squeeze(v, 0) for v in tf.split(self.policy.loc, len(self.dis_action_shape), 0)],
                [tf.squeeze(v, 0) for v in tf.split(self.policy.scale, len(self.dis_action_shape), 0)],
                tf.split(self.valid_data_mask(self.action), [1]*len(self.dis_action_shape), axis=-1)
            ):
                self.add_summary("particles/{}".format(n), tf.nn.tanh(p) if self.normalize_policy_output or self.normalize_policy_output_ else p)
                self.add_summary("particles_std/{}".format(n), pstd)
                self.add_summary("cont_action/{}".format(n), ca)

        if self.trainable and self.resample:
            with tf.name_scope("resample"):
                self.max_active = tf.get_variable("max_active_degree",
                    shape=self.policy.dis_dist.probs.shape[1:], dtype=self.policy.dis_dist.probs.dtype,
                    trainable=False, initializer=tf.zeros_initializer()
                )
                self.sum_active = tf.get_variable("sum_active_degree",
                    shape=self.policy.dis_dist.probs.shape[1:], dtype=self.policy.dis_dist.probs.dtype,
                    trainable=False, initializer=tf.zeros_initializer()
                )
                new_active = tf.maximum(self.max_active, tf.math.reduce_max(self.policy.dis_dist.probs, axis=0))
                update_max_active = tf.assign(self.max_active, new_active)

                new_active = self.sum_active + tf.math.reduce_sum(self.policy.dis_dist.probs, axis=0)
                update_sum_active = tf.assign(self.sum_active, new_active)

                self.local_update_variables.append(self.max_active)
                self.local_update_variables.append(self.sum_active)
                self.running_update_ops.append(update_max_active)
                self.running_update_ops.append(update_sum_active)

                train_flag = tf.Variable(0., name="train_flag",
                    shape=(), dtype=tf.float32, trainable=False
                )
                def update():
                    def _resample():
                        with tf.control_dependencies(self.build_resample_ops(self.policy, self.logstd, self.policy_bias, self.policy_weight)):
                            ops = [
                                tf.assign(self.max_active, tf.zeros_like(self.max_active)),
                                tf.assign(self.sum_active, tf.zeros_like(self.sum_active)),
                                tf.assign(train_flag, tf.zeros_like(train_flag))
                            ]
                            return tf.group(*ops)
                    op = tf.cond(tf.math.greater_equal(tf.assign_add(train_flag, 1.), self.resample_interval),
                        _resample, tf.no_op
                    )
                    return op
                self.train_ops.append(update)

    def build_resample_ops(self, policy, logstd, policy_bias, policy_weight):
        loc, std = policy.loc, policy.scale
        filter_op = []
        if self.identical_n_particles:
            n = self.dis_action_shape[0]
            assert(n >= self.resample)
            active_threshold = self.resample_threshold if self.resample_threshold else .05/n

            max_active = self.max_active
            avg_active = self.sum_active / tf.reduce_sum(self.sum_active, axis=1, keepdims=True)
            invalid = tf.cast(tf.where(max_active < active_threshold), tf.int32)

            indices = invalid[:,0]
            n_invalid = tf.shape(invalid, out_type=invalid.dtype)[0]
            
            if self.resample < 0:
                assert(self.resample == -1)
                cand = tf.random.categorical(tf.math.log(avg_active), n, dtype=invalid.dtype)
                choice = invalid[:,1]
                k = n
            else:
                k = min(n, self.resample)
                _, cand = tf.math.top_k(avg_active, k=k, sorted=False)
                choice = tf.random.uniform([n_invalid], 0, k, invalid.dtype)

            choice_ = tf.range(n_invalid, dtype=invalid.dtype)*k+choice
            offset = n*indices
            columns = offset+invalid[:,1]
            target_columns = offset+tf.gather_nd(cand, tf.stack([indices, choice], -1))
            assert(self.policy_weight.shape[1] == n*len(self.dis_action_shape))
            rows = tf.range(self.policy_weight.shape[0], dtype=invalid.dtype)
            invalid_mesh = tf.reshape(tf.stack(
                tf.meshgrid(rows, columns, indexing='ij'),
            axis=-1),  [-1, 2])

            target_loc = tf.gather(tf.gather(loc, cand, batch_dims=-1), indices)
            target_loc = tf.gather(tf.reshape(target_loc, [-1]), choice_)
            target_std = tf.gather(tf.gather(std, cand, batch_dims=-1), indices)
            target_std = tf.gather(tf.reshape(target_std, [-1]), choice_)
            target_logstd = tf.gather(tf.gather(logstd, cand, batch_dims=-1), indices)
            target_logstd = tf.gather(tf.reshape(target_logstd, [-1]), choice_)
            target_bias = tf.gather(policy_bias, target_columns)
            target_weight = tf.gather(policy_weight, target_columns, axis=-1)


            if self.fixed_sigma and (self.normalize_policy_output or self.normalize_policy_output_):
                target_loc_ = target_loc
                target_loc = tf.nn.tanh(target_loc_) 
                if not self.fixed_sigma:
                    target_std = tf.maximum(
                        tf.nn.tanh(target_loc_ + target_std) - target_loc,
                        target_loc - tf.nn.tanh(target_loc_ - target_std)
                    )
            if self.fixed_sigma:
                target_std = self.init_sigma
                
            noise = target_std*tf.random.uniform(tf.shape(target_loc), -1, 1, dtype=target_loc.dtype)
            noise += tf.where(
                noise < 0, -1e-4*tf.ones_like(noise), 1e-4*tf.ones_like(noise)
            )
            target_loc += noise
            

            if self.fixed_sigma and self.normalize_policy_output or self.normalize_policy_output_:
                eps = 1e-6
                target_loc = tf.math.atanh(tf.clip_by_value(target_loc, eps-1, 1-eps))
            target_logstd = tf.clip_by_value(target_logstd, -20, 2)

            target_columns_, idx, count = tf.unique_with_counts(target_columns)
            delta = tf.map_fn(
                lambda x: tf.reduce_sum(tf.cast(tf.equal(columns, x), target_bias.dtype), axis=-1),
                target_columns_, dtype=target_bias.dtype
            )
            target_bias -= tf.gather(tf.log(tf.cast(count, target_bias.dtype)+1-delta), idx)
            
            update_s = tf.scatter_nd_update(loc, invalid, target_loc)
            filter_op.append(update_s)
            if not self.fixed_sigma:
                update_std = tf.scatter_nd_update(logstd, invalid, target_logstd)
                filter_op.append(update_std)
            with tf.control_dependencies([tf.scatter_update(policy_bias, target_columns, target_bias)]):
                update_bias = tf.scatter_update(policy_bias, columns, target_bias)
            update_weight = tf.scatter_nd_update(
                policy_weight, invalid_mesh, tf.reshape(target_weight, [-1])
            )
            filter_op.append(update_bias)
            filter_op.append(update_weight)
        else:
            raise NotImplementedError
        return filter_op

    def build_policy(self, trainable, last_layer, action_shape):
        if self.identical_n_particles:
            n = self.dis_action_shape[0]
            if self.action_upper_bound is not None and self.action_lower_bound is not None :
                u, l = [1.]*len(self.action_upper_bound), [-1.]*len(self.action_lower_bound)
            else:
                u, l = self.action_upper_bound, self.action_lower_bound
            u = np.repeat(np.expand_dims(u, -1), n, -1)
            l = np.repeat(np.expand_dims(l, -1), n, -1)
            if self.normalize_policy_output_ or self.normalize_policy_output:
                loc = l + (u-l)/n*(np.expand_dims(np.arange(n), 0)+0.5)
            else:
                loc = l + (u-l)/(n-1)*np.expand_dims(np.arange(n), 0)

            if self.init_sigma:
                std = self.init_sigma
                if self.normalize_policy_output_ or self.normalize_policy_output:
                    loc_ = loc
                    loc = np.arctanh(loc)
                    std = np.maximum(
                        loc-np.arctanh(np.maximum(1e-6-1, np.subtract(loc_, std))),
                        np.arctanh(np.minimum(1-1e-6, np.add(loc_, std)))-loc
                    )
            else:
                std = (u-l)/(n-1)
                if self.normalize_policy_output_ or self.normalize_policy_output:
                    assert(n > 3)
                    loc = np.arctanh(loc)
                    std = []
                    t = np.full(loc.shape, 1)
                    for i in range(len(t)):
                        std.append([])
                        for j in range(len(t[0])):
                            d0 = loc[i][j] - loc[i][max(0,j-int(t[i][j]))]
                            d1 = loc[i][min(n-1, j+int(t[i][j]))] - loc[i][j]
                            d = max(d0, d1)
                            std[-1].append(d)

            max_sigma = None
            if max_sigma is not None:
                sigma_eps = 2e-9
                max_sigma = np.expand_dims(max_sigma, -1)
                logstd_offset = 0.5*(np.log(max_sigma)+np.log(sigma_eps))
                logstd_scale = 0.5*(np.log(max_sigma)-np.log(sigma_eps))
                logstd = (np.log(std)-logstd_offset)/logstd_scale
                inv_tanh_logstd = 0.5*(np.log(1+logstd)-np.log(1-logstd))
            else:
                logstd = np.log(std)

            loc = tf.get_variable("samples", shape=[len(self.dis_action_shape), n], dtype=tf.float32,
                trainable=self.trainable,
                initializer=tf.constant_initializer(loc)
            )
            logstd = tf.get_variable("samples_std", shape=[len(self.dis_action_shape), n], dtype=tf.float32,
                trainable=self.trainable and not self.fixed_sigma,
                initializer=tf.constant_initializer(
                    inv_tanh_logstd if max_sigma is not None else logstd
                )
            )
            self.logstd = logstd 
            if max_sigma is not None: logstd = tf.tanh(logstd)*logstd_scale+logstd_offset
        else:
            raise NotImplementedError
    
        index = 2 if last_layer.shape[1].value is None else 1
        logits = fc_layer("fc_policy", last_layer,
                    last_layer.shape[index:].num_elements(), sum(self.dis_action_shape),
                    weight_initializer=self.weight_initializer, activator=None,
                    trainable=trainable
        )
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if "fc_policy" in v.name:
                if "weight" in v.name:
                    self.policy_weight = v
                if "bias" in v.name:
                    self.policy_bias = v
                    
        if self.identical_n_particles:
            logits = tf.reshape(logits, [-1, len(self.dis_action_shape), self.dis_action_shape[0]])
        else:
            logits = tf.split(logits, self.dis_action_shape, axis=1)

        p = MixtureGaussianDistribution(logits=logits, loc=loc, scale=tf.exp(logstd), normalize_output=self.normalize_policy_output_)
        return p
