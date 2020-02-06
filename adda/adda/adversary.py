from contextlib import ExitStack

import tensorflow as tf
import tflearn
from tensorflow.contrib import slim

from adda.models import register_model_fn

@register_model_fn('adv')
def adversarial_discriminator(net, layers, scope='adversary', leaky=True, reuse=False, output_unit=2):
    if leaky:
        activation_fn = tflearn.activations.leaky_relu
    else:
        activation_fn = tf.nn.relu
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=activation_fn,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            for dim in layers:
                net = slim.fully_connected(net, dim)
            net = slim.fully_connected(net, output_unit, activation_fn=None)
    return net


