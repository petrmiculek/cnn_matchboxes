# do I need imports here?
import tensorflow as tf


class Accu:
    accu_base = tf.metrics.SparseCategoricalAccuracy()
    __name__ = 'accu'

    @classmethod
    @tf.function
    def __call__(cls, y_true, y_pred):
        """
        SparseCategoricalAccuracy metric + tweaks

        input reshaped from (Batch, 1, 1, 8) to (Batch, 8)
        prediction fails for equal probabilities
        (Had I not done this explicitly,
        argmax would output 0 and sometimes
        match the 0=background class label)
        """

        # reshape prediction
        y_pred_reshaped = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[3]])

        # make prediction fail if it is undecided (all probabilities are 1/num_classes = 0.125)
        cond = tf.expand_dims(tf.math.equal(tf.math.reduce_max(y_pred_reshaped, axis=1), 0.125), axis=1)
        y_avoid_free = tf.where(cond, 7.0, y_true)

        return cls.accu_base(y_avoid_free, y_pred_reshaped)


# class Augmentation:
#     def __init__(self):
#         self.model = tf.keras.Sequential([
#             # tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
#             tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.2, fill_mode='reflect', seed=1234),
#         ])
#
#     def __call__(self, *args, **kwargs):
#         return self.model(*args, **kwargs)


def safestr(*args):
    """Turn string into a filename
    https://stackoverflow.com/questions/7406102/create-sane-safe-filename-from-any-unsafe-string
    :param string:
    :return:
    """
    string = str(args)
    keepcharacters = (' ', '.', '_')
    return "".join(c for c in string if c.isalnum() or c in keepcharacters).rstrip().replace(' ', '_')
