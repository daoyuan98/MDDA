from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn


@register_model_fn('digitbn_encoder')
def lenet(inputs, keep_prob, scope='digitbn_encoder', is_training=True, train_adda=False, reuse=False):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            stack.enter_context(slim.arg_scope([slim.conv2d], padding='SAME'))

            net = slim.conv2d(net, 64, 5, stride=1, activation_fn=None, scope='conv1')
            layers['conv1'] = net

            net = slim.batch_norm(net, scope='norm1')
            layers['norm1'] = net

            net = tf.nn.relu(net)
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            layers['pool1'] = net

            net = slim.conv2d(net, 64, 5, stride=1, activation_fn=None, scope='conv2')
            layers['conv2'] = net

            net = slim.batch_norm(net, scope='norm2')
            layers['norm2'] = net

            net = tf.nn.relu(net)

            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            layers['pool2'] = net

            net = slim.conv2d(net, 128, 5, stride=1, activation_fn=None, scope='conv3')
            layers['conv3'] = net

            net = slim.batch_norm(net, scope='norm3')
            layers['norm3'] = net

            net = tf.nn.relu(net)
            net = tf.contrib.layers.flatten(net)
             
            # 3072
            net = slim.fully_connected(net, 3072, activation_fn=None, scope='fc1')
            layers['fc1'] = net
            net = tf.nn.relu(net)

            if is_training:
                net = tf.nn.dropout(net, keep_prob)
                layers['dropout'] = net

            # 2048
            net = slim.fully_connected(net, 2048, activation_fn=None, scope='fc2')
            layers['fc2'] = net

            if not train_adda:
                net = tf.nn.relu(net)         

    return net, layers

