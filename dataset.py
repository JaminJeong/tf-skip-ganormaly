from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import random

class Dataset:
    @staticmethod
    def shuffle(train_images, train_labels):
        assert train_images.shape[0] == train_labels.shape[0]
        train_len = train_labels.shape[0]
        shuffled_index = list(range(train_len))
        random.seed(12345)
        random.shuffle(shuffled_index)
        train_images_list = [train_images[i] for i in shuffled_index]
        train_labels_list = [train_labels[i] for i in shuffled_index]
        train_images = np.array(train_images_list)
        train_labels = np.array(train_labels_list)
        return train_images, train_labels

class MnistDataset:
    def __init__(self, test_label_value=[2, 3]):
        print("Load training and eval data from tf.keras!!")
        # Load training and eval data from tf.keras
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.test_label_value = test_label_value
        self.init()

    def init(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = \
            tf.keras.datasets.mnist.load_data()

        self.train_images, self.train_labels, = Dataset.shuffle(self.train_images, self.train_labels)

        train_images_list = []
        train_labels_list = []
        for image, label in zip(self.train_images, self.train_labels):
            if not label in self.test_label_value:
                train_images_list.append(image)
                train_labels_list.append(label)

        self.train_images = np.array(train_images_list)
        self.train_labels = np.array(train_labels_list)
        print(f"self.train_labels : {self.train_labels}")
        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)
        print(f"test_labels : {self.test_labels}")

    def get_train_data(self):
        return self.train_images, self.train_labels

    def get_test_data(self):
        return self.test_images, self.test_labels


from skimage.util import random_noise

class FashinMnishDataset:
    """
    FashinMnishDataset
    """
    def __init__(self):
        print("Load training and eval data from tf.keras!!")
        # Load training and eval data from tf.keras
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = \
            tf.keras.datasets.fashion_mnist.load_data()

        parameter_list = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']

        self.random_noise_test_data = np.array([255 * random_noise(x, mode=parameter_list[random.randint(0, len(parameter_list) - 1)]) \
                                                for x in self.test_images])

    def get_train_data(self):
        return self.train_images, self.train_labels

    def get_test_data(self):
        return self.test_images, self.test_labels

    def get_random_noise_test_data(self):
        return self.test_images, self.test_labels