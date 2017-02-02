import tensorflow as tf
import tensorflow.contrib as tc


def leaky_relu(x, alpha=0.1):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self):
        self.x_dim = 784
        self.name = 'mnist/dcgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc1 = tc.layers.fully_connected(x, 1024,
                                            weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                            weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                                            activation_fn=tf.identity
                                            )
            fc1 = leaky_relu(tc.layers.batch_norm(fc1))
            fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            return fc2, tf.sigmoid(fc2)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def loss(self, prediction, target):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target))


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 784
        self.name = 'mnist/dcgan/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            bs = tf.shape(z)[0]
            fc1 = tc.layers.fully_connected(z, 1024,
                                            weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                            weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                                            activation_fn=tf.identity)
            fc1 = tc.layers.batch_norm(fc1)
            fc1 = tf.nn.relu(fc1)
            fc2 = tc.layers.fully_connected(fc1, 784,
                                            weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                            weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                                            activation_fn=tf.sigmoid)
            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]