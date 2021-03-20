import tensorflow as tf
import pathlib
import glob
import os
import numpy as np


def get_class_weights(class_counts_train):
    # todo equation
    class_counts_train = np.array(class_counts_train)
    num_classes = len(class_counts_train)
    class_counts_sum = np.sum(class_counts_train)
    class_weights = class_counts_sum / (num_classes * class_counts_train)
    return dict(enumerate(class_weights))


def profile(x):
    return x


@profile
def get_dataset(data_dir):
    """
    Get dataset, class names and class weights

    Inspired by https://www.tensorflow.org/tutorials/load_data/images

    :param data_dir:
    :return:
    """
    const_seed = 1234
    autotune = tf.data.experimental.AUTOTUNE

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)

        # The second to last is the class-directory
        # (purpose = class-number-independent encoding)

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
        ds = ds.shuffle(buffer_size=1000, seed=const_seed)  # reshuffle_each_iteration=True
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=autotune)
        return ds

    batch_size = 64

    dataset = tf.data.Dataset.list_files(os.path.join(data_dir, '*/*.jpg'), shuffle=True)

    """ Compile a `class_names` list from the tree structure of the files """
    class_dirs, class_counts = np.unique(np.array(sorted([item.parent for item in pathlib.Path(data_dir).glob('*/*')])), return_counts=True)
    class_names = [os.path.basename(directory) for directory in class_dirs]

    """ Load images + labels, configure """
    dataset = dataset.map(process_path, num_parallel_calls=autotune)

    class_weights = get_class_weights(class_counts)  # training set only

    dataset = configure_for_performance(dataset)

    return dataset, class_names, class_weights
