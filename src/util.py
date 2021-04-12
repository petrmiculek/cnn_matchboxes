import os
import glob
import shutil
from math import log10

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

from eval_images import full_prediction_all

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
    # todo unused
    def call(self, y_true, y_pred):
        # tf.print(tf.shape(y_true), tf.shape(y_pred))
        return super(Scce, self).call(y_true, y_pred)


class Accu(tf.metrics.SparseCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # reshape prediction - keep only Batch and Class-probabilities dimensions
        y_pred_reshaped = pred_reshape(y_pred)

        # make prediction fail if it is undecided (all probabilities are 1/num_classes = 0.125)
        cond = tf.expand_dims(tf.math.equal(tf.math.reduce_max(y_pred_reshaped, axis=1), 0.125), axis=1)

        y_avoid_free = tf.where(cond, tf.cast(7, dtype=tf.int64), y_true)

        return super(Accu, self).update_state(y_avoid_free, y_pred_reshaped, sample_weight)


class Precision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_reshaped = pred_reshape(y_pred)
        y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1, 0)

        # y_true is evaluated as bool => ok as it is

        return super(Precision, self).update_state(y_true, y_pred_binary, sample_weight)


class Recall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_reshaped = pred_reshape(y_pred)
        y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1, 0)

        return super(Recall, self).update_state(y_true, y_pred_binary, sample_weight)


class F1(tfa.metrics.F1Score):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_reshaped = pred_reshape(y_pred)
        # y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1.0, 0.0)
        # y_true = tf.greater(y_true, 0)
        # threshold = tf.reduce_max(y_pred_reshaped, axis=-1, keepdims=True)
        # y_pred_extra = tf.logical_and(y_pred_reshaped >= threshold, tf.abs(y_pred_reshaped) > 1e-12)

        # tf.print(tf.shape(y_pred_reshaped), tf.shape(y_pred_binary), tf.shape(y_pred_extra))

        # tf.print(y_true)

        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), 8)
        # tf.print(tf.shape(y_pred * y_true))

        return super(F1, self).update_state(y_true, y_pred_reshaped, sample_weight)


# https://datascience.stackexchange.com/questions/48246/how-to-compute-f1-in-tensorflow
# unused
def f1_metric(y_true, y_pred):
    true_positives = tf.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), dtype=tf.float32)
    possible_positives = tf.cast(K.sum(K.round(K.clip(y_true, 0, 1))), dtype=tf.float32)
    predicted_positives = tf.cast(K.sum(K.round(K.clip(y_pred, 0, 1))), dtype=tf.float32)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    tf.print(precision, recall, f1_val)
    return f1_val


class AUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_reshaped = pred_reshape(y_pred)
        y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1, 0)

        return super(AUC, self).update_state(y_true, y_pred_binary, sample_weight)


# # not supported in eager :/
# class mAP(tf.compat.v1.metrics.average_precision_at_k):
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred_reshaped = pred_reshape(y_pred)
#         y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1, 0)
#
#         return super(mAP, self).update_state(y_true, y_pred_binary, sample_weight)


class MSELogger(tf.keras.callbacks.Callback):
    def __init__(self, freq=1):
        super().__init__()
        self._supports_tf_logs = True
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and epoch % self.freq == 0:
            base_model = self.model.layers[1]

            # validation data
            mse_val = full_prediction_all(base_model, val=True, output_location=None, show=False)

            print(f'avg_mse_val: 1e{log10(mse_val + 1):0.2g}')
            tf.summary.scalar('avg_mse_val', mse_val, step=epoch)

            # training data
            mse_train = full_prediction_all(base_model, val=False, output_location=None, show=False)
            print(f'avg_mse_train: 1e{log10(mse_train + 1):0.2g}')
            tf.summary.scalar('avg_mse', mse_train, step=epoch)


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr


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
        if training is None:
            training = tf.keras.backend.learning_phase()
        if not training:
            return images

        images = tf.image.random_contrast(images, self.contrast_range[0], self.contrast_range[1])
        images = tf.image.random_brightness(images, self.brightness)
        images = tf.image.random_hue(images, self.hue)
        images = tf.image.random_saturation(images, self.saturation_range[0], self.saturation_range[1])
        images = tf.clip_by_value(images, 0, 255)
        return images

    def get_config(self, *args, **kwargs):
        return {
            'brightness_delta': self.brightness,
            'contrast_range': self.contrast_range,
            'hue_delta': self.hue,
            'saturation_range': self.saturation_range,
        }

    # from_config() needs not to be reimplemented


def conv_dim_calc(w, k, d=1, p=0, s=1):
    """Calculate change of dimension size caused by a convolution layer

    out = (w - f + 2p) / s + 1
    f = 1 + (k - 1) * d

    :param w: width
    :param k: kernel size
    :param d: dilation rate
    :param p: padding
    :param s: stride
    :return: output size
    """
    return (w - (1 + (k - 1) * d) + 2 * p) // s + 1
