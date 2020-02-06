import os

import numpy as np
import tensorflow as tf
import pickle as pkl

import json
import base64
import cv2

head_path = "/mdda/"
from tensorflow.python.estimator.inputs.queues.feeding_queue_runner import _FeedingQueueRunner as FeedingQueueRunner

def str2image(base64_data):
    imgData = base64.b64decode(base64_data)
    nparr = np.fromstring(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

class DatasetGroup(object):
    def __init__(self, dataset_name, source_name, size=256, split_ratio=0.8, max_num=1000000, train_adda=0):
        self.dataset_name = dataset_name
        self.size = size
        self.image_shape = (size, size, 3)
        self.label_shape = ()
        self.source_name = source_name
        self.train_adda = train_adda
        self.split_ratio = split_ratio
        self.max_num = max_num
        self.random_crop = True

        self._load_datasets()

    def get_path(self, *args):
        pass

    def download(self):
        pass

    def shuffle(self, seed):
        pass

    def _load_datasets(self):
        def namelist_2_dataset(source_name_list):
            image_list = []
            json_info = {}
            source_num = len(source_name_list)
            for source_name in source_name_list:
                json_path = '{}/data/{}/{}/raw/annotations.json'.format(head_path, self.dataset_name, source_name)
                json_ =  json.loads(open(json_path).read())
                sub_json_info = json_['image_info']
                for idx, (file_name, info) in enumerate(sub_json_info.items()):
                    if idx < 34000:
                        json_info[file_name] = info

            image_list = list(json_info.keys())
            image_num  = min(self.max_num, len(image_list))
            seed = np.random.randint(0, 10000)
            np.random.seed(seed)
            np.random.shuffle(image_list)
            train_images = []
            train_labels = []

            for i in range(image_num):
                images = str2image(json_info[image_list[i]]['image_str'])
                train_images.append(cv2.resize(images, (self.image_shape[0], self.image_shape[0])))
                train_labels.append(json_info[image_list[i]]['class_id'])
            return train_images, train_labels

        if ':' in self.source_name:
            train_name_list = self.source_name.split(':')[0].split('-')
            test_name_list = self.source_name.split(':')[1].split('-')
            train_images, train_labels =  namelist_2_dataset(train_name_list)
            test_images, test_labels =  namelist_2_dataset(test_name_list)
        else:    
            source_name_list = self.source_name.split('-')
            image_list = []
            json_info = {}
            source_num = len(source_name_list)
            for source_name in source_name_list:
                json_path = '{}/data/{}/{}/raw/annotations.json'.format(head_path, self.dataset_name, source_name)
                json_ =  json.loads(open(json_path).read())
                sub_json_info = json_['image_info']
                for idx, (file_name, info) in enumerate(sub_json_info.items()):
                    if idx < 34000:
                        json_info[file_name] = info
                
            self.num_classes = len(json_['class_list'])

            image_list = list(json_info.keys())
            image_num  = min(self.max_num, len(image_list))
            np.random.seed(19)
            np.random.shuffle(image_list)

            self.n_train = int(image_num * self.split_ratio) - 1
            self.n_test = image_num - self.n_train

            train_shape = (self.n_train, ) + self.image_shape
            test_shape = (self.n_test, ) + self.image_shape

            train_images = []
            train_labels = []
            train_image_names = []

            test_images = []
            test_labels = []
            test_image_names = []

            for i in range(self.n_train):
                # images = cv2.imread(image_list[i])
                images = str2image(json_info[image_list[i]]['image_str'])
                train_images.append(cv2.resize(images, (self.image_shape[0], self.image_shape[0])))
                train_labels.append(json_info[image_list[i]]['class_id'])
                train_image_names.append(image_list[i])
  
            for i in range(self.n_train, self.n_train + self.n_test):
                # images = cv2.imread(image_list[i])
                images = str2image(json_info[image_list[i]]['image_str'])
                test_images.append(cv2.resize(images, (self.image_shape[0], self.image_shape[0])))
                test_labels.append(json_info[image_list[i]]['class_id'])
                test_image_names.append(image_list[i])

        train_images = np.array(train_images, np.float32)
        train_labels = np.array(train_labels, np.uint8)
        train_image_names = np.array(train_image_names)

        test_images = np.array(test_images, np.float32)
        test_labels = np.array(test_labels, np.uint8)
        test_image_names = np.array(test_image_names)
        

        # if self.train_adda==1 and (image_num < (30000 * source_num)):
        if self.train_adda==1:
            train_images = np.vstack((train_images, test_images))
            train_labels = np.hstack((train_labels, test_labels))
            train_image_names = np.hstack((train_image_names, test_image_names)) 

            test_images = train_images
            test_labels = train_labels
            test_image_names = train_image_names
   
        from collections import Counter
        print('Train: ', Counter(train_labels))
        print('Test  :', Counter(test_labels))
        print(self.source_name, ' train shape: ', train_images.shape, train_labels.shape)
        print(self.source_name, ' test  shape: ', test_images.shape, test_labels.shape)
        print(train_labels[30:100])
        print(test_labels[30:100])
 
        if self.dataset_name == "digit":
            train_images = (train_images - np.array([128., 128., 128.]))
            test_images = (test_images - np.array([128., 128., 128.]))

        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=False)

        self.train_image_names = train_image_names # ImageDataset(train_image_names, None, image_shape=None, label_shape=None, shuffle=False)
        self.test_image_names = test_image_names

        self.test = ImageDataset(test_images, test_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=False)


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

# def register_dataset(name):
#     def decorator(cls):
#         datasets[name] = cls
#         return cls
#     return decorator

def get_ft_dataset(source, target, solver, lr, batch_size):
    #added by songzhichao
    # path = 'pkls/' + source + '_' + target + '.pkl'
    path = os.path.join("/nfs/cold_project/iccv/data/FineTune", source+"_FT_"+target+"_"+solver+"_"+lr+"_"+batch_size)
    mat = pkl.load(open(path, 'rb+'))
    train_images = np.array(mat['image'])
    train_labels = np.array(mat['label'])
    idxs = np.random.permutation(train_images.shape[0])
    train_images = train_images[idxs]
    train_labels = train_labels[idxs]
    train = ImageDataset(train_images, train_labels,image_shape=(28,28,3),label_shape=())
    return train


# def get_dataset_guyang(name, *args, **kwargs):
#     return datasets[name](*args, **kwargs)

def get_dataset2(*args, **kwargs):
    return DatasetGroup(*args, **kwargs)

