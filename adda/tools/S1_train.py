import logging
import os
import random
import sys
from collections import deque


import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append('..')
import adda

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

@click.command()
@click.argument('dataset')
@click.argument('test_target')
@click.argument('split')
@click.argument('model')
@click.argument('root_dataset')
@click.argument('class_num', type=int)
@click.argument('exp_name')
@click.option('--gpu', default='0')
@click.option('--iterations', default=20000)
@click.option('--batch_size', default=64)
@click.option('--display', default=10)
@click.option('--lr', default=1e-4)
@click.option('--snapshot', default=5000)
@click.option('--stepsize', default=1000)
@click.option('--solver', default='sgd')
@click.option('--seed', type=int)
def main(dataset, test_target, root_dataset, exp_name, split, model, gpu, iterations, batch_size, display,
         lr, class_num, snapshot, stepsize, solver, seed):

    head_path = "/mdda"
    if root_dataset == 'digit':
        keep_prob_num = 0.9

    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    if seed is None:
        seed = 19
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)
    source_name = dataset.split(':')[0]

    output_encoder = "{}/model/{}/{}/encoder/{}_E_{}_{}_{}".format(head_path, root_dataset, source_name, source_name, source_name, model, exp_name)
    output_classifier = "{}/model/{}/{}/classifer/{}_C_{}_{}_{}".format(head_path, root_dataset, source_name, source_name, source_name, model, exp_name)

    target_encoder_scopes = []
    classifier_scopes = []
    encoder = adda.models.get_model_fn(model+'_encoder')
    classifier = adda.models.get_model_fn(model+'_classifier')
    is_train_dataset = tf.placeholder(dtype=tf.bool, shape=())
    keep_prob = tf.placeholder(dtype=tf.float32, shape=())

    if dataset == 'usps':
        dataset1 = adda.data.get_dataset2(root_dataset, dataset, size=28, split_ratio=0.784147, max_num=1000000, train_adda=False)
    else:
        dataset1 = adda.data.get_dataset2(root_dataset, dataset, size=28, split_ratio=0.73532353, max_num=1000000, train_adda=False)
    train_dataset = getattr(dataset1, 'train') 

    train_im, train_label = train_dataset.tf_ops()
    train_im = adda.models.preprocessing(train_im, encoder)
    train_im_batch, train_label_batch = tf.train.batch([train_im, train_label], batch_size=batch_size)

    im_batch = train_im_batch
    label_batch = train_label_batch

    target_encoder_scope = 'target_' + source_name + '_' + source_name
    target_encoder_scopes.append(target_encoder_scope)

    fts, layer = encoder(im_batch, keep_prob, scope=target_encoder_scope)
    classifier_scope = 'classifier_' + source_name + '_' + source_name
    classifier_scopes.append(classifier_scope)
    classifier_net, layer = classifier(fts, n_class=class_num, scope=classifier_scope)

    class_loss = tf.losses.sparse_softmax_cross_entropy(label_batch, classifier_net)
    loss = class_loss

    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.9)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var)
    step = optimizer.minimize(loss)

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    var_dict_encoder = adda.util.collect_vars(target_encoder_scopes[0])
    if not os.path.exists(output_encoder):
        os.makedirs(output_encoder)
    saver_encoder = tf.train.Saver(var_list=var_dict_encoder)
    var_dict_classifier = adda.util.collect_vars(classifier_scopes[0])
    if not os.path.exists(output_classifier):
        os.makedirs(output_classifier)
    saver_classifier = tf.train.Saver(var_list=var_dict_classifier)

    all_dir_name_list = output_encoder.split("/")
    all_dir_name = os.path.join(all_dir_name_list[0], "all_"+all_dir_name_list[1])
    if not os.path.exists(all_dir_name):
        os.makedirs(all_dir_name)
    losses = deque(maxlen=10)
    epoch_num = int(len(train_dataset)/batch_size)

    for i in range(iterations + 1):
        
        loss_val, _ = sess.run([loss, step], feed_dict={is_train_dataset:True, keep_prob:keep_prob_num})
        losses.append(loss_val)

        if i % display == 0:
            logging.info('{:20} {:10.4f}     (avg: {:10.4f})'
                         .format('Iteration {}:'.format(i),
                                 loss_val,
                                 np.mean(losses)))
        if i > 10 and i % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.1))
            logging.info('Change learning rate to: {}'.format(lr))

    snapshot_path = saver_encoder.save(sess, os.path.join(output_encoder, "encoder_{}".format(str(i))), global_step=i + 1)
    logging.info('Saved snapshot to {}'.format(snapshot_path))

    classifier_snapshot_path = saver_classifier.save(sess, os.path.join(output_classifier, "classifier_{}".format(str(i))), global_step=i + 1)
    logging.info('Saved snapshot to {}'.format(classifier_snapshot_path))

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
