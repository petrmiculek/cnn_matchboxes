# stdlib
import os
import pathlib
import glob

# external
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def weights_sklearn(class_counts):
    """Weigh classes (using sklearn) by their inverse frequency"""

    fake_ds = [np.repeat(i, cc) for i, cc in enumerate(class_counts)]
    fake_ds = np.hstack(fake_ds)
    return compute_class_weight('balanced',
                                classes=np.unique(fake_ds),
                                y=fake_ds)


def weights_inverse_freq(class_counts):
    """Weigh classes by their inverse frequency"""
    class_counts = np.array(class_counts)
    class_weights = np.sum(class_counts) / (len(class_counts) * class_counts)
    return dict(enumerate(class_weights))


def weights_effective_number(class_counts):
    """Weight classes by effective number of samples

    weights are less aggresive than inverse frequency

    Class-Balanced Loss Based on Effective Number of Samples
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
    """
    class_counts = np.array(class_counts)

    beta = (class_counts - 1) / class_counts
    effective_number = 1.0 - np.power(beta, class_counts)
    weights = (1 - beta) / effective_number
    weights = weights / np.sum(weights) * len(class_counts)

    return dict(enumerate(weights))


def get_dataset(dataset_dir, batch_size=64, weights=None):
    """
    Get dataset, class names and class weights

    Inspired by https://www.tensorflow.org/tutorials/load_data/images

    :param dataset_dir: dataset directory
    :param batch_size: batch size
    :param weights: how to weigh classes - 'none', 'inv_freq', 'eff_num'
    :return: Dataset, Class names, Class weights
    """

    if not os.path.isdir(dataset_dir):
        raise UserWarning('Invalid dataset directory:', dataset_dir)

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
        """Join input image and label"""
        label = get_label(file_path)

        # load raw data
        img = tf.io.read_file(file_path)

        img = tf.image.decode_jpeg(img, channels=3)
        return img, label

    def configure_for_performance(ds):
        """Configure dataset for effective using"""
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=512, reshuffle_each_iteration=True)
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

    if weights == 'eff_num':
        class_weights = weights_effective_number(class_counts)
    elif weights == 'inv_freq':
        class_weights = weights_inverse_freq(class_counts)
    else:
        class_weights = dict(zip(range(0, len(class_names)), np.ones(len(class_names))))

    dataset = configure_for_performance(dataset)

    return dataset, class_names, class_weights
