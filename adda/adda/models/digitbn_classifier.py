from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn


@register_model_fn('digitbn_classifier')
def lenet(inputs, n_class=10, scope='digitbn_classifier', is_training=True, reuse=False):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))

            net = slim.fully_connected(net, n_class, activation_fn=None, scope='fc3')
            layers['fc3'] = net

    return net, layers

lenet.default_image_size = 28
lenet.num_channels = 3
lenet.mean = None
lenet.bgr = False
