import logging
import os
import time
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import sys
head_path="/mdda"
sys.path.append('{}/adda'.format(head_path))
import adda

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])


@click.command()
@click.argument('target_domain')
@click.argument('split')
@click.argument('model') 
@click.argument('source_domain')
@click.argument('root_dataset')
@click.argument('exp_name')
@click.argument('image_size')
@click.argument('class_num')
@click.option('--adda_encoder', default=None)
@click.option('--train_adda', type=int, default=0)
@click.option('--ft_classifer', default=None)
def main(target_domain, split, model, source_domain, train_adda, root_dataset, exp_name, image_size, adda_encoder, class_num, ft_classifer):
    
    head_path = "/mdda"
    
    batch_size = 64
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    target_dataset = adda.data.get_dataset2(root_dataset, target_domain, size=int(image_size), split_ratio=0.73532353, max_num=10000000, train_adda=False)
    split = getattr(target_dataset, split)
 
    encoder_scope = model + '_encoder'
    classifier_scope = model + '_classifier'  

    model_fn = adda.models.get_model_fn(encoder_scope)
    classifier = adda.models.get_model_fn(classifier_scope)
    im, label = split.tf_ops()
    im = adda.models.preprocessing(im, model_fn)
    im_batch, label_batch = tf.train.batch([im, label], batch_size=batch_size)

    keep_prob = tf.placeholder(dtype=tf.float32, shape=())

    feat, encoder_layers = model_fn(im_batch, keep_prob, scope=encoder_scope, is_training=False)    
    net, layers = classifier(feat, n_class=int(class_num), scope=classifier_scope, is_training=False)

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    var_dict = adda.util.collect_vars(encoder_scope)
    restorer = tf.train.Saver(var_list=var_dict)
    weights = '{}/model/{}/{}/encoder/{}_E_{}_{}_{}'.format(head_path, root_dataset, source_domain, source_domain, target_domain, model, exp_name)
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    restorer.restore(sess, weights)

    var_dict = adda.util.collect_vars(classifier_scope)
    restorer = tf.train.Saver(var_list=var_dict)
    weights = '{}/model/{}/{}/newclassifer/{}_NC_{}_{}_{}'.format(head_path, root_dataset, source_domain, source_domain, target_domain, model, exp_name)
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    restorer.restore(sess, weights)

    class_correct = np.zeros(target_dataset.num_classes, dtype=np.int32)
    class_counts = np.zeros(target_dataset.num_classes, dtype=np.int32)
    ntest = len(split)

    gts = []
    outputs = np.zeros((ntest, 10))
    output_gts = np.zeros((ntest, ))
    index = 0

    ntest = int(len(split)/batch_size)
    for idx in range(ntest):
        predictions, gt = sess.run([net, label_batch], feed_dict={keep_prob:1.0})
        for bs_id in range(batch_size):
            outputs[index] = predictions[bs_id]
            output_gts[index] = gt[bs_id]
            index += 1

    predictions, gt = sess.run([net, label_batch], feed_dict={keep_prob:1.0})
    for bs_id in range(len(split) - batch_size * ntest):
        outputs[index] = predictions[bs_id]
        output_gts[index] = gt[bs_id]
        index += 1

    save_path = '{}/result/{}/{}'.format(head_path, target_domain, source_domain)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    output_save_path = os.path.join(save_path, 'output.npy')
    output_gts_save_path = os.path.join(save_path, 'gts.npy')
    np.save(output_save_path, outputs)
    np.save(output_gts_save_path, output_gts)

    coord.request_stop()
    coord.join(threads)
    sess.close()
    

if __name__ == '__main__':
    main()
