import tensorflow as tf
import pathlib
import glob
import os
import numpy as np


def get_dataset(data_dir):
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    def configure_for_performance(ds):
        """
        buffered prefetching = yield data without I/O blocking.
        two important methods when loading data.

        `.cache()` keeps the images in memory after they're loaded off disk during the first epoch.
        This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory,
        you can also use this method to create a performant on-disk cache.

        `.prefetch()` overlaps data preprocessing and model execution while training.

        [data performance guide](https://www.tensorflow.org/guide/data_performance#prefetching).


        :param ds:
        :return:
        """
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=autotune)
        return ds

    batch_size = 32
    img_height = 64
    img_width = 64

    list_ds = tf.data.Dataset.list_files(os.path.join(data_dir, '*/*.jpg'), shuffle=False)
    image_count = len(list_ds)

    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    """Compile a `class_names` list from the tree structure of the files."""
    data_dir_path = pathlib.Path(data_dir)
    class_names = np.array(sorted([item.name for item in data_dir_path.glob('*')]))
    num_classes = len(class_names)

    """Split train and validation:"""

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    """
    # length of datasets
    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())
    """

    """Use `Dataset.map` to create a dataset of `image, label` pairs:"""

    autotune = tf.data.experimental.AUTOTUNE

    val_as_batch_dataset = val_ds

    train_ds = train_ds.map(process_path, num_parallel_calls=autotune)
    val_ds = val_ds.map(process_path, num_parallel_calls=autotune)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    return class_names, train_ds, val_ds, val_as_batch_dataset
