from .ops import *
import numpy as np
import tensorflow_probability as tfp

def discount(val, factor, boostrap_val, normalize=False):
    result = [None]*len(val)
    v_ = boostrap_val
    for t in reversed(range(len(val))):
        v_ = val[t] + factor * v_
        result[t] = v_
    if normalize:
        result = np.subtract(result, np.mean(result))
        result = np.divide(result, np.std(result)+1e-6)
        return list(result[::-1])
    return result

def build_conv_fc_net(trainable, last_layer, net_shape,
        weight_initializer, activator, last_activator,
        init_name_index=1):
    for i in range(len(net_shape)):
        if not hasattr(net_shape[i], "__len__") or len(net_shape[i]) == 1:
            index = 2 if last_layer.shape[1].value is None else 1
            last_layer = fc_layer(
                "fc{}".format(i+init_name_index), last_layer,
                last_layer.shape[index:].num_elements(), net_shape[i],
                weight_initializer=weight_initializer,
                activator=last_activator if i+1 == len(net_shape) else activator,
                trainable=trainable
            )
        else:
            assert(len(net_shape[i]) == 4)
            last_layer = conv_layer(
                "conv{}".format(i+init_name_index), last_layer,
                # in channel (NHWC),        out channel
                last_layer.shape[-1], net_shape[i][0],
                # kernel size, stride size, padding type
                net_shape[i][1], net_shape[i][2], net_shape[i][3],
                weight_initializer=weight_initializer,
                activator=last_activator if i+1 == len(net_shape) else activator,
                trainable=trainable
            )
     
    return last_layer


def online_normalizer(trainable, shape, moving_average=False):
    mean = tf.get_variable("mean", shape=shape, dtype=tf.float32,
                        trainable=False,
                        initializer=tf.zeros_initializer())
    std = tf.get_variable("std", shape=shape, dtype=tf.float32,
                    trainable=False,
                    initializer=tf.ones_initializer())
    if not moving_average:
        count = tf.get_variable("counter", shape=(), dtype=tf.float32,
                            trainable=False,
                            initializer=tf.constant_initializer(1e-4))

    if trainable:
        if moving_average:
            def update_op(X):
                decay = tf.constant(0.9999, dtype=tf.float32)
                s = tf.cast(tf.train.get_or_create_global_step(), dtype=tf.float32)
                decay = tf.minimum(decay, (1+s)/(10+s))

                m, v = tf.nn.moments(X, axes=[0])
                new_mean = decay*mean+(1-decay)*m 
                new_std = tf.maximum(1e-6, decay*std+(1-decay)*tf.sqrt(v))
                return [(mean, new_mean), (std, new_std)]
        else:
            def update_op(X):
                batch_mean, batch_var = tf.nn.moments(X, axes=[0])
                batch_count = tf.cast(tf.shape(X)[0], count.dtype)
                delta = batch_mean - mean
                new_count = count + batch_count
                m_a = tf.math.square(std) * count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + tf.math.square(delta) * count * batch_count / new_count
                new_std = tf.maximum(1e-6, tf.sqrt(M2 / new_count))
                new_mean = mean + delta * batch_count / new_count
                return [(mean, new_mean), (std, new_std), (count, new_count)]
        return mean, std, update_op
    return mean, std, None


class MixtureGaussianDistribution(object):
    def __init__(self, logits, loc, scale, normalize_output):
        self._identical = True
        if hasattr(logits, "__len__"):
            assert(len(logits) == len(loc) and len(loc) == len(scale))
            self._identical = False
        self.logits, self.loc, self.scale = logits, loc, scale
        self.normalize_output = normalize_output
        cont_dist = tf.distributions.Normal
        if self._identical:
            self.dis_dist_ = tf.distributions.Categorical(logits=logits)
            self.dis_dist = tfp.distributions.RelaxedOneHotCategorical(1.0, logits=logits)
            self.dis_dist.probs = self.dis_dist.distribution.probs
            self.dis_dist.logits = self.dis_dist.distribution.logits
            self.sample_dist = cont_dist(loc, scale, allow_nan_stats=False)
        else:
            raise NotImplementedError
    
    def prob(self, value, name="prob"):
        with tf.name_scope(name):
            p = tf.exp(self.log_prob(value))
        return p
    
    def log_prob(self, value, name="log_prob"):
        @tf.custom_gradient
        def foo(p):
            def grad(dy):
                # the movement of particles may cause target prob to be 0 (log prob to -inf)
                # for a proper choice of hyperparameters, this barely happens
                return tf.where(tf.logical_or(tf.is_nan(dy), tf.is_inf(dy)),
                    tf.zeros_like(dy), dy
                )
            return p, grad
            
        if self._identical:
            if self.normalize_output:
                if hasattr(value, "__len__"):
                    value, value_before_tanh = value
                else:
                    value_before_tanh = tf.math.atanh(value)
            else:
                value_before_tanh = value

            p = self.sample_dist.prob(tf.expand_dims(value_before_tanh, -1))
            p = tf.reduce_sum(self.dis_dist.probs*p, axis=-1)
            p = foo(p)
            lp = tf.log(p)
            if self.normalize_output:
                lp -= 2*(np.log(2) - value_before_tanh - tf.nn.softplus(-2*value_before_tanh))
            lp = tf.reduce_sum(lp, axis=-1)
        else:
            lp = []
            for dis, cont, a in zip(self.dis_dist.policy, self.sample_dist, value):
                p = cont.prob(a)
                p = dis.probs*p
                p = tf.reduce_sum(p, axis=1)
                p = foo(p)
                lp.append(tf.log(p))
            lp = tf.add_n(lp)
        return lp

    def entropy(self, name="entropy"):
        v = self.dis_dist.logits - tf.reduce_max(self.dis_dist.logits, axis=-1, keepdims=True)
        s0 = tf.exp(v)
        s1 = tf.reduce_sum(s0, axis=-1, keepdims=True)
        p = s0 / s1
        return tf.reduce_sum(p * (tf.log(s1) - v), axis=-1)

    def sample(self, n):
        assert(n == 1)
        if self._identical:
            if self.normalize_output: # rsample
                d = self.dis_dist.sample(n)
                w = tf.reshape(d, [-1]+list(self.logits.shape[1:]))
                p = self.sample_dist.sample(tf.shape(w)[0])
                self.dis_action = tf.argmax(w, axis=-1)
                m = tf.one_hot(self.dis_action, w.shape[-1], dtype=w.dtype)
                if self.normalize_output: 
                    tanh_p = tf.nn.tanh(p)
                    @tf.custom_gradient
                    def mask2(w, p):
                        y = m*p
                        tanh_t = tf.reduce_sum(m*tanh_p, axis=-1, keepdims=True)
                        def grad(dy):
                            gap = (tanh_p-tanh_t)/tf.maximum(1e-6, 1-tanh_t**2)
                            return gap*dy, m*dy
                        return y, grad
                    s_ = mask2(w, p)
                    s_ = tf.reduce_sum(s_, -1)
                    s_ = tf.reshape(s_, [n, -1]+list(self.logits.shape[1:-1]))
                    p = tanh_p
                @tf.custom_gradient
                def mask(w, p):
                    y = m*p
                    t = tf.reduce_sum(y, axis=-1, keepdims=True)
                    def grad(dy):
                        gap = p-t
                        return gap*dy, m*dy
                    return y, grad
                sample = mask(w, p)
                sample = tf.reduce_sum(sample, -1)
                sample = tf.reshape(sample, [n, -1]+list(self.logits.shape[1:-1]))
            else:
                d = self.dis_dist_.sample(n)
                self.dis_action = w = tf.reshape(d, [-1]+list(self.logits.shape[1:-1]))
                p = self.sample_dist.sample(tf.shape(w)[0])
                mask = tf.one_hot(self.dis_action, self.logits.shape[-1], dtype=p.dtype)
                sample = tf.reduce_sum(mask*p, -1)
                sample = tf.reshape(sample, [n, -1]+list(self.logits.shape[1:-1]))
                return sample
        else:
            raise NotImplementedError
        if self.normalize_output:
            return sample, s_
        else:
            return sample
    
    def mean(self):
        if not hasattr(self, "_determinstic_action"):
            if self._identical:
                if self.normalize_output: # rsample
                    t = 1.
                    w = tf.nn.softmax(self.logits / t)
                    p = tf.expand_dims(self.sample_dist.mean(), 0)
                    self.dis_action = tf.argmax(w, axis=-1)
                    m = tf.one_hot(self.dis_action, w.shape[-1], dtype=w.dtype)

                    if self.normalize_output: p = tf.nn.tanh(p)
                    @tf.custom_gradient
                    def mask(w, p):
                        y = m*p
                        t = tf.reduce_sum(y, axis=-1, keepdims=True)
                        def grad(dy):
                            gap = p-t
                            return gap*dy, tf.reduce_sum(m*dy, 0, keepdims=True)
                        return y, grad
                    cont_action = tf.reduce_sum(mask(w, p), -1)
                else:
                    dis_action = tf.argmax(self.logits, axis=-1)
                    dis_action = tf.transpose(dis_action, [1, 0])
                    cont_action = tf.batch_gather(self.loc, dis_action)
                    cont_action = tf.transpose(cont_action, [1, 0])
            else:
                raise NotImplementedError
                dis_action = [tf.argmax(p.logits, axis=1) for p in self.policy]
                cont_action = []
                for d, a in zip(self.sample_dist, dis_action):    
                    a = tf.gather(d.loc, a)
                    cont_action.append(a)
                cont_action = tf.stack(cont_action, axis=1)
            self._determinstic_action = cont_action
        return self._determinstic_action
