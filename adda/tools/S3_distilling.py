import logging
import os
import random
import sys
head_path="/mdda"
sys.path.append('{}/adda'.format(head_path))
from collections import deque
import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import adda
from adda.data import ImageDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

@click.command()
@click.argument('source')
@click.argument('target')
@click.argument('model')
@click.argument('root_dataset')
@click.argument('class_num', type=int)
@click.argument('encoder_exp_name')
@click.argument('classifer_exp_name')
@click.argument('exp_name')
@click.argument('result_image_path')
@click.argument('adda_encoder')
@click.argument('ft_classifer')
@click.argument('lr', type=float)
@click.argument('stepsize', type=int)
@click.argument('snapshot', type=int)
@click.argument('solver', default='sgd')
@click.argument('iterations', type=int)
@click.argument('batch_size', type=int)
@click.argument('display', type=int)
def main(source, target, model, root_dataset, class_num, encoder_exp_name, classifer_exp_name, 
         exp_name, result_image_path, adda_encoder, ft_classifer, lr, stepsize, snapshot, solver, iterations, batch_size, display):
    
    image_size = 28
    keep_prob_num = 0.9
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    
    seed = 19
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)

    output_classifier = "{}/model/{}/{}/newclassifer/{}_NC_{}_{}_{}".format(head_path, root_dataset, source, source, target, model, exp_name)

    encoder = adda.models.get_model_fn(model+'_encoder')
    classifier = adda.models.get_model_fn(model+'_classifier')
    is_train_dataset = tf.placeholder(dtype=tf.bool, shape=())
    keep_prob = tf.placeholder(dtype=tf.float32, shape=())
    
    train_labels_np_path = os.path.join(result_image_path, "wgan_label.npy")
    train_images_np_path = os.path.join(result_image_path, "wgan_image.npy")
    train_images = np.load(train_images_np_path)
    train_labels = np.load(train_labels_np_path)
    train_image_shape = train_images[0].shape
    train_label_shape = train_labels[0].shape
    train_dataset = ImageDataset(train_images, train_labels, image_shape=train_image_shape, label_shape=train_label_shape, shuffle=False)

    train_im, train_label = train_dataset.tf_ops()
    train_im = adda.models.preprocessing(train_im, encoder)
    train_im_batch, train_label_batch = tf.train.batch([train_im, train_label], batch_size=batch_size)

    encoder_scope = model + '_encoder'
    adda_encoder_scope = 'adda_' + encoder_scope
    classifier_scope = model + '_classifier'  

    fts, layer = encoder(train_im_batch, keep_prob, scope=encoder_scope, is_training=False)

    classifier_net, layer = classifier(fts, n_class=class_num, scope=classifier_scope, is_training=True)
    class_loss = tf.losses.sparse_softmax_cross_entropy(train_label_batch, classifier_net)

    loss = class_loss 

    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var, 0.5)

    update_ops = adda.util.collect_vars(classifier_scope)
    step = optimizer.minimize(loss, var_list=update_ops)
    
    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    
    # install model
    var_dict = adda.util.collect_vars(encoder_scope)
    restorer = tf.train.Saver(var_list=var_dict)
    encoder_path = '{}/model/{}/{}/encoder/{}_E_{}_{}_{}'.format(head_path, root_dataset, source, source, target, model, encoder_exp_name)
    if os.path.isdir(encoder_path):
        encoder_weights = tf.train.latest_checkpoint(encoder_path)
    restorer.restore(sess, encoder_weights)

    classifier_var_dict = adda.util.collect_vars(classifier_scope)
    classifier_restorer = tf.train.Saver(var_list=classifier_var_dict)
    classifier_weights = '{}/model/{}/{}/classifer/{}_C_{}_{}_{}'.format(head_path, root_dataset, source, source, source, model, classifer_exp_name)    
    if os.path.isdir(classifier_weights):
        classifier_weights = tf.train.latest_checkpoint(classifier_weights)
    classifier_restorer.restore(sess, classifier_weights)

    var_dict_classifier = adda.util.collect_vars(classifier_scope)
    if not os.path.exists(output_classifier):
        os.makedirs(output_classifier)
    saver_classifier = tf.train.Saver(var_list=var_dict_classifier)
    saver = tf.train.Saver(var_list=var_dict_classifier)
    
    losses = deque(maxlen=10)
    bar = range(iterations + 1)
     
    epoch_num = int(len(train_dataset)/batch_size)

    for i in bar:
        
        loss_val, _ = sess.run([loss, step], feed_dict={keep_prob:keep_prob_num})
        losses.append(loss_val)

        if i % display == 0:
            logging.info('{:20} {:10.4f}     (avg: {:10.4f})'
                         .format('Iteration {}:'.format(i),
                                 loss_val,
                                 np.mean(losses)))
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.2))
            logging.info('Changed learning rate to {:.0e}'.format(lr))


    classifier_snapshot_path = saver_classifier.save(sess, os.path.join(output_classifier, "classifier_{}".format(str(i + 1))), global_step=i + 1)
    logging.info('Saved snapshot to {}'.format(classifier_snapshot_path))

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()

