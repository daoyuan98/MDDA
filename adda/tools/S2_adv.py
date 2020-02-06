import logging
import os
import sys
import random
from collections import deque

import click
import numpy as np
import tensorflow as tf

sys.path.append('..')

import adda
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'

@click.command()
@click.argument('source')
@click.argument('target')
@click.argument('model')
@click.argument('output_encoder')
@click.argument('output_classifier')
@click.argument('iterations')
@click.argument('batch_size')
@click.argument('display')
@click.argument('disc_lr')
@click.argument('encoder_lr')
@click.argument('snapshot')
@click.argument('solver')
@click.argument('class_num')
@click.argument('image_size')
@click.argument('root_dataset')
@click.argument('source_exp_name')
@click.argument('target_exp_name')
@click.argument('classfier_exp_name')
@click.argument('adversary_layers')
@click.argument('stepsize')
def main(source, target, model, output_encoder, output_classifier, iterations, batch_size, display, disc_lr, encoder_lr,
         snapshot, solver, class_num, image_size, root_dataset, source_exp_name, target_exp_name, classfier_exp_name,
         adversary_layers, stepsize):

    head_path = "/mdda"
    stepsize = int(stepsize)
    encoder_lr = float(eval(encoder_lr))
    adversary_layers = eval(adversary_layers)
    iterations = int(iterations)
    batch_size = int(batch_size)
    display = int(display)
    disc_lr = float(disc_lr)
    encoder_lr = float(encoder_lr)
    snapshot = int(snapshot)
    class_num = int(class_num)
    Lambda = 10.

    target_update_list = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2', 'norm1', 'norm2', 'norm3']

    if root_dataset == 'digit':
        keep_prob_num = 0.9
    print('root_dataset: ', root_dataset)

    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    seed = 19
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)

    logging.info('Adapting {} -> {}'.format(source, target))
    if source == 'usps':
        dataset1 = adda.data.get_dataset2(root_dataset, source, size=28, split_ratio=0.784147, max_num=1000000, train_adda=False)
    else:
        dataset1 = adda.data.get_dataset2(root_dataset, source, size=28, split_ratio=0.73532353, max_num=1000000, train_adda=False)
    train_dataset = getattr(dataset1, 'train')
        
    target_dataset = adda.data.get_dataset2(root_dataset, target, size=28, split_ratio=0.73532353, max_num=1000000, train_adda=False)
    test_dataset = getattr(target_dataset, 'train')

    source_im, source_label = train_dataset.tf_ops()
    target_im, target_label = test_dataset.tf_ops()
    model_fn = adda.models.get_model_fn(model+'_encoder')

        
    keep_prob = tf.placeholder(dtype=tf.float32, shape=())

    source_im = adda.models.preprocessing(source_im, model_fn)
    target_im = adda.models.preprocessing(target_im, model_fn)
    source_im_batch, source_label_batch = tf.train.batch(
        [source_im, source_label], batch_size=batch_size)
    target_im_batch, target_label_batch = tf.train.batch(
        [target_im, target_label], batch_size=batch_size)

    target_encoder_scope_name = 'target_' + source + '_' + target
    adversary_scope_name = source + '_' + target + '_adv'

    source_ft, _ = model_fn(source_im_batch, keep_prob, is_training=True, train_adda=True, scope='source')
    target_ft, _ = model_fn(target_im_batch, keep_prob, is_training=True, train_adda=True, scope=target_encoder_scope_name)

    source_ft = tf.reshape(source_ft, [-1, int(source_ft.get_shape()[-1])])
    target_ft = tf.reshape(target_ft, [-1, int(target_ft.get_shape()[-1])])
    adversary_ft = tf.concat([source_ft, target_ft], 0)

    source_adversary_label = tf.ones([tf.shape(source_ft)[0], 1], tf.int32)
    target_adversary_label = tf.zeros([tf.shape(target_ft)[0], 1], tf.int32)

    adversary_label = tf.concat([source_adversary_label, target_adversary_label], 0)

    output_unit = 1
    adversary_logits = adda.adversary.adversarial_discriminator(adversary_ft, adversary_layers, scope=adversary_scope_name, leaky=True, output_unit=output_unit)

    # variable collection
    source_vars = adda.util.collect_vars('source')
    target_vars = adda.util.collect_vars(target_encoder_scope_name)
    adversary_vars = adda.util.collect_vars(adversary_scope_name)

    target_update_list = ["{}/{}".format(target_encoder_scope_name, v) for v in target_update_list]

    if len(adversary_layers) >= 1:
        adv_update_list_0 = ["{}/fully_connected".format(adversary_scope_name)]
    adv_update_list = ["{}/fully_connected_{}".format(adversary_scope_name, v) for v in range(len(adversary_layers)-1)]
    adv_update_list = adv_update_list_0 + adv_update_list
    adv_update_dict = {}
    for adv in adv_update_list:
        for key, value in adda.util.collect_vars(adv).items():
            adv_update_dict[key] = value

    target_update_list = list(target_vars.values())
    adv_update_list = list(adv_update_dict.values())

    disc_lr_var = tf.Variable(disc_lr, name='disc_learning_rate', trainable=False)
    encoder_lr_var = tf.Variable(encoder_lr, name='encoder_learning_rate', trainable=False)

    disc_real, disc_fake = tf.split(adversary_logits, 2, axis=0)  # real->source fake->target
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[batch_size, 2048],
        minval=0.,
        maxval=1.
    )
    fake_data = target_ft
    real_data = source_ft
    differences = fake_data - real_data
    interpolates_ft = real_data + (alpha * differences)
    gradients = tf.gradients(adda.adversary.adversarial_discriminator(interpolates_ft, adversary_layers, scope=adversary_scope_name, leaky=True, output_unit=1, reuse=True), [interpolates_ft])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    disc_cost += Lambda * gradient_penalty
    if solver == 'adam':
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=encoder_lr,
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=target_update_list)
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=disc_lr,
            beta1=0.5,
            beta2=0.9
        ).minimize(disc_cost, var_list=adv_update_list)
    elif solver == "sgd":
        gen_train_op = tf.train.MomentumOptimizer(encoder_lr_var, 0.9).minimize(gen_cost, var_list=target_update_list)
        disc_train_op = tf.train.MomentumOptimizer(disc_lr_var, 0.9).minimize(disc_cost, var_list=adv_update_list)

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    # restore weights
    source_weights = '{}/model/{}/{}/encoder/{}_E_{}_{}_{}'.format(head_path, root_dataset, source, source, source, model, source_exp_name)
    target_weights = '{}/model/{}/{}/encoder/{}_E_{}_{}_{}'.format(head_path, root_dataset, source, source, source, model, target_exp_name)
    print("source weights: ", source_weights)
    print("target weights: ", target_weights)
    if os.path.isdir(source_weights):
        source_weights = tf.train.latest_checkpoint(source_weights)
    logging.info('Restoring weights from {}:'.format(source_weights))
    logging.info('    Restoring source model:')
    if os.path.isdir(target_weights):
        target_weights = tf.train.latest_checkpoint(target_weights)
    logging.info('Restoring weights from {}:'.format(target_weights))
    logging.info('    Restoring source model:')

    for src, tgt in source_vars.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    source_restorer = tf.train.Saver(var_list=source_vars)
    source_restorer.restore(sess, source_weights)
    
    logging.info('    Restoring target model:')
    for src, tgt in target_vars.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    target_restorer = tf.train.Saver(var_list=target_vars)
    target_restorer.restore(sess, target_weights)

    adversary_saver = tf.train.Saver(var_list=adversary_vars)

    if not os.path.exists(output_encoder):
        os.makedirs(output_encoder)

    gen_costs = deque(maxlen=10)
    disc_costs = deque(maxlen=10)

    bar = range(iterations + 1)

    if not os.path.exists(output_encoder):
        os.makedirs(output_encoder)
    saver_encoder = tf.train.Saver(var_list=target_vars)
    if not os.path.exists(output_classifier):
        os.makedirs(output_classifier)
    saver_classifier = tf.train.Saver(var_list=adversary_vars)

    for i in bar:
        
        gen_cost_val, _ = sess.run([gen_cost, gen_train_op], feed_dict={keep_prob: keep_prob_num})
        gen_costs.append(gen_cost_val)
        disc_cost_val, _ = sess.run([disc_cost, disc_train_op], feed_dict={keep_prob: keep_prob_num})
        disc_costs.append(disc_cost_val)

        if i % display == 0:
            logging.info('{:8} gen cost: {:10.4f}     (avg: {:10.4f})    disc cost: {:10.4f}     (avg: {:10.4f})'
                         .format(i, gen_cost_val, np.mean(gen_costs), disc_cost_val, np.mean(disc_costs)))
        
    snapshot_path = saver_encoder.save(sess, os.path.join(output_encoder, "encoder_{}".format(str(i+1))), global_step=i + 1)
    logging.info('Saved snapshot to {}'.format(snapshot_path))
    classifier_snapshot_path = saver_classifier.save(sess, os.path.join(output_classifier, "classifier_{}".format(str(i+1))), global_step=i + 1)
    logging.info('Saved snapshot to {}'.format(classifier_snapshot_path))


    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
