import tensorflow as tf
import numpy as np
import csv
from collections import defaultdict
import pathlib
import PIL

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# class Label:
#     def __init__(self, label_name, x, y, img_name, img_w, img_h):
#         pass

img_dims = dict()  # make use of


def load_labels(path):
    labels = defaultdict(lambda: defaultdict(list))  # file_name, categories, list of (x, y)
    global img_dims

    with open(path + os.sep + "labels.csv") as f:  # os.curdir + os.sep + path + os.sep +
        csv_data = csv.reader(f, delimiter=',')
        for line in csv_data:
            label_name, x, y, img_name, img_w, img_h = line
            full_img_name = path + os.sep + img_name

            img = labels[full_img_name]
            img[label_name].append((x, y))

            img_dims[full_img_name] = (img_w, img_h)  # overwrites old value, risky

    return labels


def names_and_labels(path):
    labels = load_labels(path)
    names = []
    labels_out = []

    for file in labels:
        names.append(file)
        file_labels = labels[file]
        labels_out.append(file_labels['corner-top'])

    return names, labels_out


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_width, img_height])


def parse_function(filename, label):
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [64, 64])
    return resized_image, label


def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


# https://cs230.stanford.edu/blog/datapipeline/

data_dir = 'sirky'
(img_width, img_height) = (4032, 3024)
batch_size = 16

names, labels = names_and_labels(data_dir)

# labels = dict()
print('\n'.join((str(i) for i in labels)))

dataset = tf.data.Dataset.from_tensor_slices((names, labels))
# dataset = dataset.shuffle(len(filenames))
dataset = dataset.map(parse_function)  # , num_parallel_calls=4
# dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
