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


def lr_scheduler(epoch, lr, start=10, decay=-0.10):
    """

    https://keras.io/api/callbacks/learning_rate_scheduler/

    :param epoch:
    :param lr:
    :param start:
    :param decay:
    :return:
    """
    if epoch < start:
        return lr
    else:
        return lr * tf.math.exp(decay)


