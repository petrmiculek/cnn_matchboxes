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
            img[label_name].append((x * scale, y * scale))

            img_dims[full_img_name] = (img_w, img_h)  # overwrites old value, risky

    return labels


def names_and_labels(path):
    labels = load_labels(path)
    names = []
    labels_out = []

    for file in labels:
        names.append(file)
        labels_out.append(labels[file])

    return names, labels_out


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_width * scale, img_height * scale])


def parse_function(filename, label):
    image_string = tf.read_file(filename)

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


# def load_images(path):
#     for file in os.walk(os.curdir + os.sep + path):
#         if not file.endswith('.jpg'):
#             continue
#
#         img = tf.io.read_file(file)
#         yield decode_img(img)


def process_path(path):
    img = tf.io.read_file(path)
    img = decode_img(img)
    str_path = str(path)
    try:
        str_path = str_path.split('\'')[1]
        print(str_path)
    except:
        print(str_path)
        str_path = 'sirky/20201020_113911.jpg'
    label_data = labels[str_path]

    return img, label_data


# https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control
data_dir = 'sirky'
# os.chdir('sirky')  # idea for no prefix work
scale = 1  # 0.25
(img_width, img_height) = (4032 * scale, 3024 * scale)
batch_size = 16

names, labels = names_and_labels(data_dir)

dataset = tf.data.Dataset.from_tensor_slices((names, labels))
# dataset = dataset.shuffle(len(filenames))
dataset = dataset.map(parse_function, num_parallel_calls=4)
# dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
