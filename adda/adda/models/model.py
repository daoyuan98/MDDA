import logging

import numpy as np
import tensorflow as tf

models = {}

def register_model_fn(name):
    def decorator(fn):
        models[name] = fn
        fn.range = None
        fn.mean = None
        fn.bgr = False
        return fn
    return decorator

def get_model_fn(name):
    return models[name]

def preprocessing(inputs, model_fn):
    inputs = tf.cast(inputs, tf.float32)
    if model_fn.range is not None:
        logging.info('Scaling images to range {}.'.format(model_fn.range))
        inputs = model_fn.range * inputs
    inputs = tf.image.resize_images(inputs, [28, 28])
    return inputs

RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
def rgb2gray(image):
    return tf.reduce_sum(tf.multiply(image, tf.constant(RGB2GRAY)), 2, keepdims=True)

def gray2rgb(image):
    return tf.multiply(image, tf.constant(RGB2GRAY))
