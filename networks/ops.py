import tensorflow as tf
import numpy as np

def clip_by_value_with_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    # if the difference between x and l or u is smaller than the precision,
    # the following may cause the result to be 0 or 2*u
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)
    
def lstm_layer(name, X, in_channels, out_filters, batch_size=None, sequence_length_tensor=None,
               initializer=tf.truncated_normal_initializer(0.0, 0.01)):
    if len(X.shape) != 2 or X.shape[1] != in_channels:
        X = tf.reshape(X, [-1, in_channels])
    with tf.variable_scope(name):
        if sequence_length_tensor is None:
            sequence_length_tensor = tf.placeholder(tf.float32, [None], name="seq_len")
        cell = tf.nn.rnn_cell.LSTMCell(out_filters, state_is_tuple=True, initializer=initializer)

        c = tf.placeholder(tf.float32, [None, cell.state_size.c], name="state_c")
        h = tf.placeholder(tf.float32, [None, cell.state_size.h], name="state_h")
        init_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        if batch_size is None:
            if sequence_length_tensor is None:
                batch_size = 1
            else:
                with tf.name_scope("batch_size"):
                    batch_size = tf.shape(sequence_length_tensor)[0]

        with tf.variable_scope("seq_batch"):
            X = tf.reshape(X, [batch_size, -1, in_channels])
        Y, state = tf.nn.dynamic_rnn(cell, X,
                                     initial_state=init_state,
                                     sequence_length=sequence_length_tensor,
                                     time_major=False)
    return tf.reshape(Y, [-1, out_filters]), (state, (c, h))


def conv_layer(name, X, in_channels, out_filters, ksize, stride,
               activator,
               # tf.variance_scaling_initializer
               # tf.truncated_normal_initializer(0.0, 0.01)
               # tf.orthogonal_initializer
               # tf.glorot_uniform_initializer()
               weight_initializer,
               bias_initializer=tf.constant_initializer(0.0),
               padding="SAME", trainable=True):
    if not hasattr(ksize, "__len__"):
        ksize = [ksize, ksize]
    if not hasattr(stride, "__len__"):
        stride = [stride, stride]
    if len(X.shape) == 2:
        X = tf.reshape(X, [-1, 1, X.shape[1], 1])
        in_channels = 1
    else:
        assert(len(X.shape) == 4)
    with tf.variable_scope(name):
        if callable(weight_initializer):
            try:
                weight_initializer = weight_initializer()
            except:
                pass
        w_shape = [ksize[0], ksize[1], in_channels, out_filters]
        w = tf.get_variable("weight", w_shape, tf.float32,
                            weight_initializer, trainable=trainable)
        Y = tf.nn.conv2d(X, w, [1, stride[0], stride[1], 1], padding)
        if bias_initializer is not None:
            if callable(bias_initializer):
                try:
                    bias_initializer = bias_initializer()
                except:
                    pass
            b = tf.get_variable("bias", [out_filters], tf.float32,
                                bias_initializer,
                                trainable=trainable)
            Y = tf.add(Y, b)
            if activator is not None:
                Y = activator(Y)
    return Y

def fc_layer(name, X, in_channels, out_filters,
             activator,
             # tf.variance_scaling_initializer
             # tf.truncated_normal_initializer(0.0, 0.01)
             # tf.orthogonal_initializer(),
             # tf.glorot_uniform_initializer()
             weight_initializer,
             bias_initializer=tf.constant_initializer(0.0),
             trainable=True):
    if X.shape.ndims != 2 or X.shape[1] != in_channels:
        X = tf.reshape(X, [-1, in_channels])
    with tf.variable_scope(name):
        if callable(weight_initializer):
            if weight_initializer == tf.orthogonal_initializer:
                weight_initializer = tf.orthogonal_initializer(np.sqrt(2) if activator == tf.nn.relu else 1)
            else:
                try:
                    weight_initializer = weight_initializer()
                except:
                    pass
        w = tf.get_variable("weight", [in_channels, out_filters], tf.float32,
                            weight_initializer,
                            trainable=trainable)
        Y = tf.matmul(X, w)
        if bias_initializer is not None:
            if callable(bias_initializer):
                try:
                    bias_initializer = bias_initializer()
                except:
                    pass
            b = tf.get_variable("bias", [out_filters], tf.float32,
                                bias_initializer,
                                trainable=trainable)
            Y = tf.add(Y, b)
            if activator is not None:
                Y = activator(Y)
    return Y
