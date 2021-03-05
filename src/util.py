import tensorflow as tf

"""
Model-related utility code
(as opposed to non-model-related utility code in src_util)
"""


class Accu(tf.metrics.SparseCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # reshape prediction
        y_pred_reshaped = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[3]])

        # make prediction fail if it is undecided (all probabilities are 1/num_classes = 0.125)
        cond = tf.expand_dims(tf.math.equal(tf.math.reduce_max(y_pred_reshaped, axis=1), 0.125), axis=1)
        y_avoid_free = tf.where(cond, 7.0, y_true)

        return super(Accu, self).update_state(y_avoid_free, y_pred_reshaped, sample_weight)


def lr_scheduler(epoch, lr, start=10, end=150, decay=-0.10):
    """

    https://keras.io/api/callbacks/learning_rate_scheduler/

    :param epoch:
    :param lr:
    :param start:
    :param end:
    :param decay:
    :return:
    """
    if epoch < start:
        return lr
    elif epoch > end:
        return lr
    else:
        return lr * tf.math.exp(decay)


class RandomColorDistortion(tf.keras.layers.Layer):
    """Apply multiple color-related augmentations

    Adapted from:
    https://github.com/GoogleCloudPlatform/practical-ml-vision-book/blob/master/06_preprocessing/06e_colordistortion.ipynb

    efficiency of chained operations (jpg -> float, aug, float -> jpg)
    """
    def __init__(self,
                 brightness_delta=0.2,
                 contrast_range=(0.5, 1.5),
                 hue_delta=0.2,
                 saturation_range=(0.75, 1.25),
                 **kwargs):
        super(RandomColorDistortion, self).__init__(**kwargs)
        self.brightness = brightness_delta
        self.contrast_range = contrast_range
        self.hue = hue_delta
        self.saturation_range = saturation_range

    def call(self, images, training=None):
        if not training:
            return images

        # contrast = tf.random.uniform((1,), self.contrast_range[0], self.contrast_range[1])[0]

        # brightness = tf.random.uniform((1,), self.brightness_range[0], self.brightness_range[1])[0]

        # hue = tf.random.uniform((1,), self.hue_range[0], self.hue_range[1])[0]
        # saturation = tf.random.uniform((1,), self.saturation_range[0], self.saturation_range[1])[0]

        # tf.print(contrast, brightness, hue)

        images = tf.image.random_contrast(images, self.contrast_range[0], self.contrast_range[1])
        images = tf.image.random_brightness(images, self.brightness)
        images = tf.image.random_hue(images, self.hue)
        images = tf.image.random_saturation(images, self.saturation_range[0], self.saturation_range[1])
        images = tf.clip_by_value(images, 0, 255)
        return images


def print_both(output_file_path):
    f = open(output_file_path, 'w')  # where to close
    print_orig = print

    def print_inner(*args):
        print_orig(*args)
        print_orig(*args, file=f)

    return print_inner


class DuplicateStream(object):
    """Make stream double-ended, outputting to stdout and a file

    http://www.tentech.ca/2011/05/stream-tee-in-python-saving-stdout-to-file-while-keeping-the-console-alive/
    Based on https://gist.github.com/327585 by Anand Kunal

    pray for Py3 functionality
    """

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None  # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name  # Could also be a property
        return getattr(self, '__methodmissing__')

    def __methodmissing__(self, *args, **kwargs):
        # Emit method call to the log copy
        callable2 = getattr(self.stream2, self.__missing_method_name)
        callable2(*args, **kwargs)

        # Emit method call to stdout (stream 1)
        callable1 = getattr(self.stream1, self.__missing_method_name)
        return callable1(*args, **kwargs)
