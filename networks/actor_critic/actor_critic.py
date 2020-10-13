import tensorflow as tf
import numpy as np
from ..ops import *
from ..utils import build_conv_fc_net, online_normalizer

class ActorCriticNetwork(object):
    def __init__(self, trainable, state_shape, action_shape,
                 common_net_shape, critic_net_shape, actor_net_shape,
                 weight_initializer=tf.glorot_uniform_initializer, activator=tf.nn.relu6,
                 normalize_state=False, clip_state=False,
                 normalize_value=False, clip_value=False,
                 normalize_advantage=False, clip_advantage=False,
                 critic_regularizer=None, actor_regularizer=None,
                 entropy_beta=None, value_loss_coef=0.5,
                 log=True, **kwargs):
        self.trainable = trainable
        self.random_action = self.trainable
        self.state_shape = list(state_shape) if hasattr(state_shape, "__len__") else [state_shape]
        self.action_shape = action_shape 

        self.common_net_shape = common_net_shape
        self.critic_net_shape = critic_net_shape
        self.actor_net_shape = actor_net_shape

        self.weight_initializer = weight_initializer
        self.activator = activator

        self.normalize_state = normalize_state
        self.clip_state = clip_state
        self.normalize_value = normalize_value
        self.clip_value = clip_value
        self.normalize_advantage = normalize_advantage
        self.clip_advantage = clip_advantage

        self.critic_regularizer = critic_regularizer
        self.actor_regularizer = actor_regularizer

        self.entropy_beta = entropy_beta
        self.value_loss_coef = value_loss_coef
        self.init_ops, self.train_ops, self.running_update_ops = [], [], []
        self.local_update_variables = []
        self.summaries = {}
        self.log = log

    # compatible for LSTM operators
    def reset(self):
        pass
    def reset_running_state(self):
        pass
    def reset_training_state(self):
        pass
    
    def add_summary(self, key, val, val_wrapper=None):
        if not self.log: return
        if val is None:
            if key not in self.summaries:
                self.summaries[key] = {}
        else:
            summary_scope = "summary/"
            with tf.name_scope(summary_scope):
                if val_wrapper is not None:
                    val = val_wrapper(val)
                if "/" in key:
                    key = key.split("/")
                    if key[0] not in self.summaries:
                        self.summaries[key[0]] = {}
                    if len(key) > 1:
                        self.summaries[key[0]]["/".join(key[1:])] = val
                    else:
                        self.summaries[key[0]] = val
                else:
                    self.summaries[key] = val
        
    def init(self):    
        self.state = tf.placeholder(tf.float32, [None] + self.state_shape, name="state")
        if self.normalize_state or self.clip_state:
            self.state_normalizer = self.build_state_normalizer_op()
            x = tf.stop_gradient(self.state_normalizer(self.state))
        else:
            x = self.state

        self.common_net = self.build_common_net_op()
        x = self.common_net(x)

        self.actor_net = self.build_actor_net_op()
        self.policy = self.actor_net(x)
        with tf.name_scope("action"):
            self.action = self.build_action_sampler(self.random_action, self.policy)
            if self.action is not None:
                self.add_summary("action/action", self.valid_data_mask(self.action))

        if self.trainable:
            self.action_hist = self.setup_action_hist_tensor(self.action_shape)

        self.critic_net = self.build_critic_net_op()
        self.normalized_value = self.critic_net(x)

        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
        
        if self.normalize_value or self.clip_value and self.normalize_value is not None:
            self.value_denormalizer = self.build_value_denormalizer_op()
            self.value = self.value_denormalizer(self.normalized_value)
        else:
            self.value = self.normalized_value

        if self.trainable:
            self.advantage = self.setup_advantage_tensor()
            self.value_target = self.setup_value_target_tensor()
            if self.normalize_state:
                with tf.name_scope("state_normalizer/update"):
                    self.setup_state_normalizer(self.state_normalizer_update_op, self.valid_data_mask(self.state))
            if self.normalize_value:
                with tf.name_scope("value_denormalizer/update"):
                    self.setup_value_normalizer(self.value_normalizer_update_op, self.valid_data_mask(self.value_target))

            if self.value_target is not None:
                if self.value_target.shape.ndims == 2 and self.value_target.shape[1] == 1:
                    self.value_target = tf.squeeze(self.value_target, axis=1)
                self.add_summary("model/value_target", self.valid_data_mask(self.value_target))
                if self.normalize_value:
                    with tf.variable_scope("normalize_value_target"):
                        self.normalized_value_target = (self.value_target - self.value_mean) / self.value_std
                        self.add_summary("model/value_target_normalized", self.valid_data_mask(self.value_target))
                else:
                    self.normalized_value_target = self.value_target
                if self.clip_value:
                    with tf.name_scope("clip_value_target"):
                        self.normalized_value_target = self.value_clipping_op(self.normalized_value_target)
                self.normalized_value_target = tf.stop_gradient(self.normalized_value_target)
                    
            with tf.variable_scope("value_loss"):
                self.value_loss = self.build_value_loss(self.normalized_value, self.normalized_value_target)
                self.add_summary("loss/value_loss", self.value_loss)
                if self.value_loss_coef != 1.0:
                    self.value_loss *= tf.constant(self.value_loss_coef, dtype=tf.float32, name="{}".format(self.value_loss_coef))
                if self.critic_regularizer and self.critic_vars:
                    with tf.name_scope("regularizer"):
                        value_reg = self.critic_regularizer * tf.add_n([tf.nn.l2_loss(v) for v in self.critic_vars])
                    self.value_loss += value_reg

            if self.entropy_beta:
                with tf.variable_scope("policy_entropy"):
                    self.policy_entropy = self.build_policy_entropy(self.policy)
                    self.add_summary("loss/policy_entropy", self.policy_entropy)
            else:
                self.policy_entropy = None

            if self.advantage is not None:
                self.add_summary("model/advantage", self.advantage)
            if self.normalize_advantage and self.advantage is not None:
                with tf.variable_scope("normalize_advantage"):
                    self.advantage_mean, v = tf.nn.moments(self.valid_data_mask(self.advantage), axes=[0])
                    self.advantage_std = tf.sqrt(v)
                    self.normalized_advantage = (self.advantage - self.advantage_mean) / (self.advantage_std + 1e-8)
                    self.add_summary("model/advantage_normalized", self.valid_data_mask(self.normalized_advantage))
            else:
                self.normalized_advantage = self.advantage
            if self.normalized_advantage is not None and self.clip_advantage:
                with tf.name_scope("clip_advantage"):
                    with tf.name_scope("clip_range"):
                        if hasattr(self.clip_advantage, "__len__"):
                            assert(len(self.clip_advantage) == 2)
                            lim0 = tf.constant(min(self.clip_advantage[0], self.clip_advantage[1]), dtype=tf.float32, name="lb_{}".format(self.clip_advantage[0]))
                            lim1 = tf.constant(max(self.clip_advantage[0], self.clip_advantage[1]), dtype=tf.float32, name="ub_{}".format(self.clip_advantage[0]))
                        else:
                            lim0 = tf.constant(-self.clip_advantage, dtype=tf.float32, name="lb_{}".format(-self.clip_advantage))
                            lim1 = tf.constant(self.clip_advantage, dtype=tf.float32, name="ub_{}".format(self.clip_advantage))
                    self.normalized_advantage = tf.clip_by_value(self.normalized_advantage, lim0, lim1)
            if self.normalized_advantage is not None:
                self.normalized_advantage = tf.stop_gradient(self.normalized_advantage)
            with tf.variable_scope("policy_loss"):
                self.policy_loss = self.build_policy_loss(self.policy, self.action_hist, self.normalized_advantage)
                self.add_summary("loss/policy_loss", self.policy_loss)
                if self.policy_entropy is not None:
                    self.policy_loss -= self.entropy_beta * self.policy_entropy
                if self.actor_regularizer and self.actor_vars:
                    with tf.name_scope("regularizer"):
                        policy_reg = self.actor_regularizer * tf.add_n([tf.nn.l2_loss(v) for v in self.actor_vars])
                    self.policy_loss += policy_reg

            with tf.variable_scope("loss"):
                self.loss = self.policy_loss + self.value_loss
                self.add_summary("loss/loss", self.loss)

    # def setup_batch_size_tensor(self, state):
    #     return tf.cast(tf.shape(state)[0], tf.float32)
    
    def setup_action_hist_tensor(self, action_shape):
        return tf.placeholder(tf.float32, [None] + list(action_shape), name="action_hist")    

    def setup_value_target_tensor(self):
        return tf.placeholder(tf.float32, shape=[None], name="value_target")

    def setup_advantage_tensor(self):
        return tf.placeholder(tf.float32, shape=[None], name="advantage")

    def build_common_net(self, trainable, last_layer, common_net_shape):
        if common_net_shape:
            return build_conv_fc_net(trainable, last_layer, common_net_shape,
                weight_initializer=self.weight_initializer,
                activator=self.activator, last_activator=self.activator
            )
        return last_layer
    
    def build_value(self, trainable, last_layer, critic_net_shape):
        return build_conv_fc_net(trainable, last_layer, [_ for _ in critic_net_shape]+[1],
            weight_initializer=self.weight_initializer,
            activator=self.activator, last_activator=None
        )

    def build_policy_net(self, trainable, last_layer, actor_net_shape):
        if actor_net_shape:
            return build_conv_fc_net(trainable, last_layer, actor_net_shape,
                weight_initializer=self.weight_initializer,
                activator=self.activator, last_activator=self.activator
            )
        return last_layer
    
    def build_policy(self, trainable, last_layer, action_shape):
        raise NotImplementedError
    
    def build_state_normalizer_op(self):
        def state_normalizer(x):
            if self.normalize_state:
                self.state_mean, self.state_std, self.state_normalizer_update_op = self.init_state_normalizer()
                self.add_summary("model/state_mean", self.state_mean)
                self.add_summary("model/state_std", self.state_std)
                with tf.variable_scope("normalize_state"):
                    x = (x - self.state_mean) / self.state_std
                    self.add_summary("model/state_normalized", self.valid_data_mask(x))
            if self.clip_state:
                with tf.name_scope("clip_state"):
                    with tf.name_scope("clip_range"):
                        if hasattr(self.clip_state, "__len__"):
                            assert(len(self.clip_state) == 2)
                            lim0 = tf.constant(min(self.clip_state[0], self.clip_state[1]), dtype=tf.float32, name="lb_{}".format(self.clip_state[0]))
                            lim1 = tf.constant(max(self.clip_state[0], self.clip_state[1]), dtype=tf.float32, name="ub_{}".format(self.clip_state[0]))
                        else:
                            lim0 = tf.constant(-self.clip_state, dtype=tf.float32, name="lb_{}".format(-self.clip_state))
                            lim1 = tf.constant(self.clip_state, dtype=tf.float32, name="ub_{}".format(self.clip_state))
                    x = tf.clip_by_value(x, lim0, lim1)
            return x
        return tf.make_template("state_normalizer", state_normalizer)

    def build_value_denormalizer_op(self):
        def value_normalizer(x):
            if self.normalize_value:
                self.value_mean, self.value_std, self.value_normalizer_update_op = self.init_value_normalizer()
                self.add_summary("model/value_mean", self.value_mean)
                self.add_summary("model/value_std", self.value_std)
                self.add_summary("model/value_normalized", self.valid_data_mask(x))
            if self.clip_value:
                if hasattr(self.clip_value, "__len__"):
                    assert(len(self.clip_value) == 2)
                    lim0 = min(self.clip_value[0], self.clip_value[1])
                    lim1 = max(self.clip_value[0], self.clip_value[1])
                else:
                    lim0 = -self.clip_value
                    lim1 = self.clip_value
                self.value_clipping_op = lambda x: tf.clip_by_value(x, lim0, lim1)
                with tf.name_scope("clip_value"):
                    x = clip_by_value_with_gradient(x, lim0, lim1)
            if self.normalize_value:
                with tf.variable_scope("denormalize_value"):
                    x = x * self.value_std + self.value_mean
                    self.add_summary("model/value", self.valid_data_mask(x))
            return x
        return tf.make_template("value_denormalizer", value_normalizer)


    def build_common_net_op(self):
        def common_net(x):
            x = self.build_common_net(self.trainable, self.input_state, self.common_net_shape)
            return x
        return tf.make_template("common", common_net) if self.common_net_shape else lambda x:x

    def build_critic_net_op(self):
        def critic_net(x):
            x = self.build_value(self.trainable, x, self.critic_net_shape)
            if x is not None:
                if x.shape.ndims == 2 and x.shape[1] == 1:
                    x = tf.squeeze(x, axis=1)
            return x
        return tf.make_template("critic", critic_net)

    def build_actor_net_op(self):
        def actor_net(x):
            x = self.build_policy_net(self.trainable, x, self.actor_net_shape)
            p = self.build_policy(self.trainable, x, self.action_shape)
            return p
        return tf.make_template("actor", actor_net)

    def build_action_sampler(self, trainable, policy):
        raise NotImplementedError
    
    def build_action_log_prob(self, policy, action):
        lp = policy.log_prob(action)
        if len(lp.shape) > 1:
            assert(len(lp.shape) == 2)
            lp = tf.reduce_sum(lp, axis=1)
        # FIXME Hack! prevent -inf
        return lp

    def build_value_loss(self, value, value_target):
        raise NotImplementedError
        
    def build_policy_loss(self, policy, action_hist, advantage):
        raise NotImplementedError

    def build_policy_entropy(self, policy):
        raise NotImplementedError

    def run(self, sess, state):
        raise NotImplementedError
    
    def train(self, sess, optimizer, ops, *args, **kwargs):
        raise NotImplementedError
    
    def init_state_normalizer(self):
        return online_normalizer(self.trainable, self.state.shape[1:], moving_average=True)

    def setup_state_normalizer(self, update_op, state):
        ref_value = update_op(state)
        # running online normalizer
        # self.running_update_ops.extend([
        #    tf.assign(ref, value) for ref, value in ref_value
        # ])
        # moving average
        for ref, value in ref_value:
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(ref, value))
        for ref, _ in ref_value:
            self.local_update_variables.append(ref)

    def init_value_normalizer(self):
        self.value_scale = tf.get_variable("scale", trainable=False,
                shape=[], dtype=tf.float32,
                initializer=tf.constant_initializer(1.0)
        )
        self.value_offset = tf.get_variable("offset", trainable=False,
                shape=[], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0)
        )
        self.normalized_value = self.normalized_value*self.value_scale + self.value_offset
        return online_normalizer(self.trainable, [], moving_average=True)
    
    def setup_value_normalizer(self, update_op, value_target):
        with tf.name_scope("value_normalizer_update"):
            ref_value = update_op(value_target)
        for ref, new_val in ref_value:
            if ref == self.value_mean: new_mean = new_val
            if ref == self.value_std: new_std = new_val
        renormalization = [
            tf.assign(self.value_offset, (self.value_mean-new_mean)/new_std),
            tf.assign(self.value_scale, self.value_std/new_std)
        ]
        with tf.control_dependencies(renormalization):
            for ref, value in ref_value:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(ref, value))
            for ref, _ in ref_value:
                self.local_update_variables.append(ref)
        self.local_update_variables.append(self.value_scale)
        self.local_update_variables.append(self.value_offset)

    def valid_data_mask(self, x):
        return x
    
    def _run(self, sess, ops, feed_dict, running_update=True):
        running_update = running_update and self.trainable and self.running_update_ops
        if running_update:
            ops.extend(self.running_update_ops)
        if callable(getattr(sess, "run_step_fn", None)):
            result = sess.run_step_fn(lambda step_context: \
                step_context.session.run(ops, feed_dict)
            )
        else:
            result = sess.run(ops, feed_dict)
        if running_update:
            result, ops = result[:-len(self.running_update_ops)], ops[:-len(self.running_update_ops)]
        return [_[0] if hasattr(_, "__len__") else _ for _, __ in zip(result, ops)]

    def _train(self, sess, optimizer, ops, feed_dict):
        training_ops = [
            self.loss, self.policy_loss, self.value_loss
        ]
        if self.policy_entropy is not None:
            training_ops.append(self.policy_entropy)

        if hasattr(optimizer, "__len__"):
            training_ops.extend(optimizer)
        else:
            training_ops.append(optimizer)
        if hasattr(ops, "__len__"):
            training_ops.extend(ops)
        else:
            training_ops.append(ops)
        result = sess.run(training_ops, feed_dict)
        loss = (
            result[training_ops.index(self.loss)],
            None if self.policy_entropy is None else result[training_ops.index(self.policy_entropy)],
            result[training_ops.index(self.policy_loss)],
            result[training_ops.index(self.value_loss)]
        )
        return loss, result[-len(ops) if hasattr(ops, "__len__") else -1:] if ops else []
    