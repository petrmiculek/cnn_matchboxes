import os
import glob
import shutil
import tensorflow as tf

"""
Model-related utility code
(as opposed to non-model-related utility code in src_util)
"""


def get_checkpoint_path(path='/tmp/model_checkpoints'):

    files = glob.glob(os.path.join(path, '*'))
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)

    return os.path.join(path, 'checkpoint')


def pred_reshape(y_pred):
    return tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[3]])


class Scce(tf.losses.SparseCategoricalCrossentropy):
    def call(self, y_true, y_pred):
        tf.print(tf.shape(y_true), y_pred)
        return super(Scce, self).call(y_true, y_pred)


class Accu(tf.metrics.SparseCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # reshape prediction - keep only Batch and Class-probabilities dimensions
        y_pred_reshaped = pred_reshape(y_pred)

        # make prediction fail if it is undecided (all probabilities are 1/num_classes = 0.125)
        cond = tf.expand_dims(tf.math.equal(tf.math.reduce_max(y_pred_reshaped, axis=1), 0.125), axis=1)
        y_avoid_free = tf.where(cond, 7.0, y_true)

        return super(Accu, self).update_state(y_avoid_free, y_pred_reshaped, sample_weight)


class Precision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_reshaped = pred_reshape(y_pred)
        y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1, 0)  # lambda: 1, lambda: 0

        # y_true is evaluated as bool => ok as it is

        return super(Precision, self).update_state(y_true, y_pred_binary, sample_weight)


class Recall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_reshaped = pred_reshape(y_pred)
        y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1, 0)  # lambda: 1, lambda: 0

        return super(Recall, self).update_state(y_true, y_pred_binary, sample_weight)




def lr_scheduler(epoch, lr, start=10, end=150, decay=-0.10):
    """Exponential learning rate decay

    https://keras.io/api/callbacks/learning_rate_scheduler/

    :param epoch: Current epoch number
    :param lr: current learning rate
    :param start: first epoch to start LR scheduling
    :param end: last epoch of LR scheduling (constant LR after)
    :param decay: Decay rate
    :return: New learning rate
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

    maybe low efficiency of chained operations (jpg -> float, aug, float -> jpg)
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

        images = tf.image.random_contrast(images, self.contrast_range[0], self.contrast_range[1])
        images = tf.image.random_brightness(images, self.brightness)
        # images = tf.image.random_hue(images, self.hue)  # todo off for now
        images = tf.image.random_saturation(images, self.saturation_range[0], self.saturation_range[1])
        images = tf.clip_by_value(images, 0, 255)
        return images

    def get_config(self, *args, **kwargs):
        # return super(RandomColorDistortion, self).get_config(*args, **kwargs)  # baseline
        return {
            'brightness_delta': self.brightness,
            'contrast_range': self.contrast_range,
            'hue_delta': self.hue,
            'saturation_range': self.saturation_range,
        }

    # from_config() needs not to be overriden=reimplemented
