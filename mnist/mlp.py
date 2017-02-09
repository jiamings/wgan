import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.1):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self):
        self.x_dim = 784
        self.name = 'mnist/mlp/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 28, 28, 1])
            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(tc.layers.batch_norm(conv2))
            conv2 = tcl.flatten(conv2)
            fc1 = tc.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(tc.layers.batch_norm(fc1))
            fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def loss(self, prediction, target):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target))


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 784
        self.name = 'mnist/mlp/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            fc = z
            fc = tcl.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tcl.batch_norm
            )
            fc = leaky_relu(fc)
            fc = tcl.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tcl.batch_norm
            )
            fc = leaky_relu(fc)
            fc = tcl.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tcl.batch_norm
            )
            fc = leaky_relu(fc)
            fc = tc.layers.fully_connected(
                fc, 784,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]