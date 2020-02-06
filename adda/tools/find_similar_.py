import logging
import os
import time
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
import pickle as pkl
sys.path.append('../../')
import adda

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'

def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])


@click.command()
@click.argument('source')
@click.argument('target')
@click.argument('model') # lenet encoder
@click.argument('input_source_encoder')
@click.argument('input_target_encoder')
@click.argument('input_discriminator')
@click.argument('result_image_path')
@click.argument('image_size')
@click.argument('root_dataset')
def main(source, target, model, input_source_encoder, input_target_encoder, input_discriminator, result_image_path, image_size, root_dataset):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    if source == 'usps':    
        dataset1 = adda.data.get_dataset2(root_dataset, source, size=28, split_ratio=0.784147, max_num=1000000, train_adda=False)
    else:
        dataset1 = adda.data.get_dataset2(root_dataset, source, size=28, split_ratio=0.73532353, max_num=1000000, train_adda=False)
    source_dataset = getattr(dataset1, 'train')
    
    target_dataset = adda.data.get_dataset2(root_dataset, target, size=28, split_ratio=0.73532353, max_num=1000000, train_adda=False)
    target_dataset = getattr(target_dataset, 'train')


    keep_prob = 1.0
    encoder = adda.models.get_model_fn(model+"_encoder")

    source_im, source_label = source_dataset.tf_ops(capacity=32)
    source_im = adda.models.preprocessing(source_im, encoder)
    source_im_batch, source_label_batch = tf.train.batch([source_im, source_label], batch_size=1)

    target_im, target_label = target_dataset.tf_ops(capacity=32)
    target_im = adda.models.preprocessing(target_im, encoder)
    target_im_batch, target_label_batch = tf.train.batch([target_im, target_label], batch_size=1)

    source_net, source_layers = encoder(source_im_batch, keep_prob, scope="source_encoder", is_training=False)
    target_net, target_layers = encoder(target_im_batch, keep_prob, scope="target_encoder", is_training=False)

    adversary_scope_name = 'adv'
    
    source_adversary_logits = adda.adversary.adversarial_discriminator(
        source_net, [3000, 2000, 1000], scope=adversary_scope_name, leaky=True, output_unit=1
    )
    target_adversary_logits = adda.adversary.adversarial_discriminator(
        target_net, [3000, 2000, 1000], scope=adversary_scope_name, leaky=True, output_unit=1, reuse=True
    )

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    # restore source encoder
    var_dict = adda.util.collect_vars("source_encoder")
    restorer = tf.train.Saver(var_list=var_dict)
    weights = input_source_encoder
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Evaluating Source Encoder {}'.format(weights))
    restorer.restore(sess, weights)

    # restore target encoder
    var_dict = adda.util.collect_vars("target_encoder")
    restorer = tf.train.Saver(var_list=var_dict)
    weights = input_target_encoder
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Evaluating Target Encoder {}'.format(weights))
    restorer.restore(sess, weights)

    # restore adversary
    adv_var_dict = adda.util.collect_vars(adversary_scope_name)
    adv_restorer = tf.train.Saver(var_list=adv_var_dict)
    adv_weights = input_discriminator
    if os.path.isdir(adv_weights):
        adv_weights = tf.train.latest_checkpoint(adv_weights)
    logging.info('Evaluating Discriminator {}'.format(adv_weights))
    adv_restorer.restore(sess, adv_weights)

    # Generate target output
    logging.info('calculating target...')
    ntest = len(target_dataset)
    target_discs = np.zeros((ntest, ))
    for i in range(ntest):
        target_sample_disc = sess.run([target_adversary_logits])
        target_discs[i] = target_sample_disc[0]
    ave_target = np.mean(target_discs)
    print(target_discs[:100], ave_target)

    # Generate Source output
    logging.info('calculating source...')
    ntest = len(source_dataset)
    source_discs = np.zeros((ntest,))
    disc_costs = np.zeros((ntest,))
    images = np.zeros((ntest, int(image_size), int(image_size), 3))
    labels = np.zeros((ntest,))
    for i in tqdm(range(ntest)):
        cur_disc_cost, image, label = sess.run([source_adversary_logits, source_im_batch, source_label_batch])
        source_discs[i] = cur_disc_cost
        disc_costs[i] = np.abs(cur_disc_cost - ave_target)
        images[i] = image
        labels[i] = label
    print(source_discs[:100])

    sorted_disc_costs = sorted(disc_costs.tolist(), reverse=False) # from small to large ones
    len_sorted = len(sorted_disc_costs)
    save_len_sorted = int(0.5 * len_sorted)
    loss_bound = sorted_disc_costs[save_len_sorted]
    idxs = np.where(disc_costs < loss_bound)
    save_images = images[idxs]
    save_labels = labels[idxs]
    save_costs = disc_costs[idxs]
    print(save_images.shape, save_labels.shape,save_costs.shape)
    print(loss_bound)
    print(save_costs)

    if not os.path.exists(result_image_path):
        os.makedirs(result_image_path)
    save_label_np_path = os.path.join(result_image_path, "wgan_label.npy")
    save_image_np_path = os.path.join(result_image_path, "wgan_image.npy")
    save_loss_np_path =  os.path.join(result_image_path, "wgan_loss.npy")
    save_source_disc_np_path = os.path.join(result_image_path, "source_disc.npy")
    save_target_disc_np_path =  os.path.join(result_image_path, "target_disc.npy")

    print("save label path: ", save_label_np_path)
    np.save(save_loss_np_path,  save_costs)
    np.save(save_image_np_path, save_images)
    np.save(save_label_np_path, save_labels)
    np.save(save_source_disc_np_path, source_discs)
    np.save(save_target_disc_np_path, target_discs)

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
