import tensorflow as tf
import pathlib
import glob
import os
import numpy as np


def get_class_weights(class_counts_train):
    num_classes = len(class_counts_train)
    class_counts_sum = np.sum(class_counts_train)
    class_counts_train = class_counts_sum / (num_classes * class_counts_train)
    return dict(zip(range(0, num_classes), class_counts_train))


def get_dataset(data_dir):
    const_seed = 1234

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        # (purpose = class-number-independent encoding)
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(tf.cast(one_hot, dtype='uint8'))

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])
        # return img

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000, seed=const_seed)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=autotune)
        return ds

    batch_size = 32
    img_height, img_width = 32, 32

    list_ds = tf.data.Dataset.list_files(os.path.join(data_dir, '*/*.jpg'), shuffle=False)
    image_count = len(list(list_ds))  # total (train + validation)

    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False, seed=const_seed)

    """Compile a `class_names` list from the tree structure of the files."""
    data_dir_path = pathlib.Path(data_dir)
    class_names = np.array(sorted([item.name for item in data_dir_path.glob('*')]))
    num_classes = len(class_names)  # total (train + validation)

    """Split train and validation:"""
    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    autotune = tf.data.experimental.AUTOTUNE

    val_as_batch_dataset = val_ds

    # map to labels, etc
    train_ds = train_ds.map(process_path, num_parallel_calls=autotune)
    val_ds = val_ds.map(process_path, num_parallel_calls=autotune)

    _, class_counts = np.unique(np.array(list(train_ds), dtype='object')[:, 1], return_counts=True)
    class_weights = get_class_weights(class_counts)  # training set only

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    return class_names, train_ds, val_ds, val_as_batch_dataset, class_weights
