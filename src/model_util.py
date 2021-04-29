# stdlib

# external
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

# local
from eval_images import full_prediction_all
from general import exp_form

"""
Model-related utility code
(as opposed to non-model-related utility code in src_util)
"""


def pred_reshape(y_pred):
    """Fix for prediction dimensions: (B, 1, 1, C) -> (B, C)

    B = batch size
    C = output channels (8)

    :param y_pred: prediction tensor
    :return: squeezed prediction tensor
    """
    # return tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[3]])
    y_pred = tf.squeeze(y_pred, axis=2)
    y_pred = tf.squeeze(y_pred, axis=1)
    return y_pred


class Accu(tf.metrics.SparseCategoricalAccuracy):
    """Calculate sample-level accuracy metric

    fixes:
    prediction shape
    make prediction fail if all predicted values equal

    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        # reshape prediction - keep only Batch and Class-probabilities dimensions
        y_pred_reshaped = pred_reshape(y_pred)

        # make prediction fail if it is undecided (all probabilities are 1/num_classes = 0.125)
        cond = tf.expand_dims(tf.math.equal(tf.math.reduce_max(y_pred_reshaped, axis=1), 0.125), axis=1)

        y_avoid_free = tf.where(cond, tf.cast(7, dtype=tf.int64), y_true)

        return super(Accu, self).update_state(y_avoid_free, y_pred_reshaped, sample_weight)


class Precision(tf.keras.metrics.Precision):
    """Calculate sample-level PR-value metric

    prediction reshaped
    prediction treated as binary (0 = background, 1 = keypoint)
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_reshaped = pred_reshape(y_pred)
        y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1, 0)

        # y_true is evaluated as bool => ok as it is

        return super(Precision, self).update_state(y_true, y_pred_binary, sample_weight)


class Recall(tf.keras.metrics.Recall):
    """Calculate sample-level Recall metric

    prediction reshaped
    prediction treated as binary (0 = background, 1 = keypoint)
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_reshaped = pred_reshape(y_pred)
        y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1, 0)

        return super(Recall, self).update_state(y_true, y_pred_binary, sample_weight)


class F1(tfa.metrics.F1Score):
    """Unused"""

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


class AUC(tf.keras.metrics.AUC):
    """Calculate PR-value metric

    prediction reshaped
    prediction treated as binary (0 = background, 1 = keypoint)
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_reshaped = pred_reshape(y_pred)
        y_pred_binary = tf.where(tf.argmax(y_pred_reshaped, axis=1) > 0, 1, 0)

        return super(AUC, self).update_state(y_true, y_pred_binary, sample_weight)


class MSELogger(tf.keras.callbacks.Callback):
    """Calculate full image prediction MSE metric

    results printed and logged to TensorBoard
    """

    def __init__(self, freq=1):
        super().__init__()
        self._supports_tf_logs = True
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and epoch % self.freq == 0:
            base_model = self.model.layers[1]

            pix_mse_val, dist_mse_val, count_mae_val = full_prediction_all(base_model, val=True, output_location=None,
                                                                           show=False)
            pix_mse_train, dist_mse_train, count_mae_train = full_prediction_all(base_model, val=False,
                                                                                 output_location=None, show=False)

            print('pix_mse:', exp_form(pix_mse_train), exp_form(pix_mse_val))
            print('dist_mse:', exp_form(dist_mse_train), exp_form(dist_mse_val))
            print('count_mae: {:0.2g} {:0.2g}'.format(count_mae_train, count_mae_val))

            tf.summary.scalar('pix_mse_train', pix_mse_train, step=epoch)
            tf.summary.scalar('dist_mse_train', dist_mse_train, step=epoch)
            tf.summary.scalar('count_mae_train', count_mae_train, step=epoch)

            tf.summary.scalar('pix_mse_val', pix_mse_val, step=epoch)
            tf.summary.scalar('dist_mse_val', dist_mse_val, step=epoch)
            tf.summary.scalar('count_mae_val', count_mae_val, step=epoch)


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Log learning rate to TensorBoard"""

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

    # from_config() does not need to be reimplemented
