import tensorflow as tf
import pathlib
import glob
import os
import numpy as np


def get_class_weights(class_counts_train):
    """

    :param class_counts_train:
    :return:
    """
    class_counts_train = np.array(class_counts_train)
    num_classes = len(class_counts_train)
    class_counts_sum = np.sum(class_counts_train)
    class_weights = class_counts_sum / (num_classes * class_counts_train)
    return dict(enumerate(class_weights))


# def profile(x):
#     return x


def get_dataset(dataset_dir, batch_size=64, use_weights=False):
    """
    Get dataset, class names and class weights

    Inspired by https://www.tensorflow.org/tutorials/load_data/images

    :param use_weights:
    :param dataset_dir:
    :param batch_size:
    :return:
    """
    assert os.path.isdir(dataset_dir)

    autotune = tf.data.experimental.AUTOTUNE

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)

        # The second to last is the class-directory
        # purpose = class-number-independent encoding

        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(tf.cast(one_hot, dtype='uint8'))

    def process_path(file_path):
        label = get_label(file_path)

        # load raw data
        img = tf.io.read_file(file_path)

        img = tf.image.decode_jpeg(img, channels=3)
        return img, label

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1024)  # reshuffle_each_iteration=True
        ds = ds.batch(batch_size)  # drop_remainder=True
        ds = ds.prefetch(buffer_size=autotune)
        return ds

    dataset = tf.data.Dataset.list_files(os.path.join(dataset_dir, '*/*.jpg'), shuffle=True)

    """ Get class names and counts from the tree structure of the files """
    class_dirs, class_counts = np.unique(np.array(sorted(
        [item.parent for item in pathlib.Path(dataset_dir).glob('*/*')])), return_counts=True)

    class_names = [os.path.basename(directory) for directory in class_dirs]

    """ Load images + labels, configure """
    dataset = dataset.map(process_path, num_parallel_calls=autotune)

    if use_weights:
        class_weights = get_class_weights(class_counts)  # unused for validation dataset
    else:
        class_weights = dict(zip(range(0, len(class_names)), np.ones(len(class_names))))

    dataset = configure_for_performance(dataset)

    return dataset, class_names, class_weights
