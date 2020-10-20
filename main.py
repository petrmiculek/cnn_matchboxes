import tensorflow as tf
import numpy as np
import csv
from collections import defaultdict

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# class Label:
#     def __init__(self, label_name, x, y, img_name, img_w, img_h):
#         pass


def load_labels(path):
    labels = defaultdict(defaultdict)  # categories
    img_dims = dict()  # make use of

    with open(os.curdir + os.sep + path + os.sep + "labels.csv") as f:  # can only iterate once
        csv_data = csv.reader(f, delimiter=',')
        for line in csv_data:
            label_name, x, y, img_name, img_w, img_h = line

            # if labels[label_lame] is None:
            #     labels[label_name] = dict()
            category = labels[label_name]

            # if category[img_name] is None:
            #     (labels[label_name])[img_name] =

            category[img_name] = (x, y)

            img_dims[img_name] = (img_w, img_h)  # overwrites old value, risky

    return labels


if __name__ == '__main__':
    print(tf.__version__)
    data_dir = 'sirky'
    (img_height, img_width) = (4032, 3024)
    batch_size = 16

    labels = load_labels(data_dir)

    # TF 2.3 and above :(
    # tf.data.Dataset
    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="training",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)
    #
    # val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)



    # validation_generator = test_datagen.flow_from_directory(
    #     validation_dir,
    #     target_size=(150, 150),
    #     batch_size=batch_size,
    #     class_mode='categorical')
