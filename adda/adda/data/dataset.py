import os

import numpy as np
import tensorflow as tf
import pickle as pkl

import json
import base64
import cv2

# to be compatible.
from tensorflow.python.estimator.inputs.queues.feeding_queue_runner import _FeedingQueueRunner as FeedingQueueRunner

def str2image(base64_data):
    imgData = base64.b64decode(base64_data)
    nparr = np.fromstring(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

class DatasetGroup(object):

    ntrain = 25000
    ntest = 9000

    seed = 19

    def __init__(self, name, path=None, download=False):
        self.name = name
        if path is None:
            path = os.path.join(os.getcwd(), 'data')
        self.path = path
        if download:
            self.download()

    def get_path(self, *args):
        return os.path.join(self.path, self.name, *args)

    def download(self):
        """Download the dataset(s).

        This method only performs the download if necessary. If the dataset
        already resides on disk, it is a no-op.
        """
        pass

    def shuffle(self, seed):
        pass


    def _load_datasets(self, train_adda, json_path, dataset_name):
        json_info = json.loads(open(json_path).read())['image_info']

        json_keys = list(json_info.keys())
        np.random.seed(19)
        json_keys = np.random.choice(json_keys, self.n_train + self.n_test)

        train_shape = (self.n_train, ) + self.image_shape
        test_shape = (self.n_test, ) + self.image_shape

        train_images = np.zeros(train_shape)
        train_labels = np.zeros((self.n_train,), dtype=np.uint8)

        test_images = np.zeros(test_shape)
        test_labels = np.zeros((self.n_test,), dtype=np.uint8)

        for i in range(self.n_train):
            train_images[i] = str2image(json_info[json_keys[i]]['image_str'])
            train_labels[i] = json_info[json_keys[i]]['class_id']

        for i in range(self.n_train, self.n_train + self.n_test):
            test_images[i - self.n_train] = str2image(json_info[json_keys[i]]['image_str'])
            test_labels[i - self.n_train] = json_info[json_keys[i]]['class_id']

        if train_adda:
            train_images = np.vstack((train_images, test_images))
            train_labels = np.hstack((train_labels, test_labels))


        from collections import Counter
        print('Train: ', Counter(train_labels))
        print('Test  :', Counter(test_labels))

        print(test_labels[:50])

        print(dataset_name, ' train shape: ', train_images.shape, train_labels.shape)
        print(dataset_name, ' test  shape: ', test_images.shape, test_labels.shape)

        train_images = train_images / 255.
        test_images = test_images / 255.

        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

        self.test = ImageDataset(test_images, test_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)


class ImageDataset(object):

    def __init__(self, images, labels, image_shape=None, label_shape=None,
                 shuffle=False):
        self.images = images
        self.labels = labels
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.shuffle = shuffle

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        inds = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(inds)
        for ind in inds:
            yield self.images[ind], self.labels[ind]

    def feed(self, im, label, epochs=None):
        epochs_elapsed = 0
        while epochs is None or epochs_elapsed < epochs:
            for entry in self:
                yield {im: entry[0], label: entry[1]}
            epochs_elapsed += 1

    def tf_ops(self, capacity=32):
        im = tf.placeholder(tf.float32, shape=self.image_shape)
        label = tf.placeholder(tf.int32, shape=self.label_shape)
        if self.image_shape is None or self.label_shape is None:
            shapes = None
        else:
            shapes = [self.image_shape, self.label_shape]
        queue = tf.FIFOQueue(capacity, [tf.float32, tf.int32], shapes=shapes)
        enqueue_op = queue.enqueue([im, label])
        fqr = FeedingQueueRunner(queue, [enqueue_op],
                                 feed_fns=[self.feed(im, label).__next__])
        tf.train.add_queue_runner(fqr)
        return queue.dequeue()


class FilenameDataset(object):

    def tf_ops(self, capacity=32):
        im, label = tf.train.slice_input_producer(
            [tf.constant(self.images), tf.constant(self.labels)],
            capacity=capacity,
            shuffle=False)
        im = tf.read_file(im)
        im = tf.image.decode_image(im, channels=3)
        return im, label


datasets = {}

def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def get_ft_dataset(source, target, solver, lr, batch_size):
    path = os.path.join("/nfs/project/iccv/data/FineTune", source+"_FT_"+target+"_"+solver+"_"+lr+"_"+batch_size)
    mat = pkl.load(open(path, 'rb+'))
    train_images = np.array(mat['image'])
    train_labels = np.array(mat['label'])
    idxs = np.random.permutation(train_images.shape[0])
    train_images = train_images[idxs]
    train_labels = train_labels[idxs]
    train = ImageDataset(train_images, train_labels,image_shape=(28,28,3),label_shape=())
    return train


def get_dataset(name, *args, **kwargs):
    return datasets[name](*args, **kwargs)
