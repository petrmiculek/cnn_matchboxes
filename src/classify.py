# -*- coding: utf-8 -*-
"""
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG

21. 12.
zkus znovu projit tutorial na fine-tuning, pripadne grad cam


"""
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import datetime

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from datasets import get_dataset
from show_results import visualize_results, predict_full_image
from class_activation_map import single_image
import models

from IPython.display import Image, display
import matplotlib.cm as cm

print(f'{tf.__version__=}')

if len(tf.config.list_physical_devices('GPU')) == 0:
    print('no available GPUs')
    sys.exit(0)

# from google.colab import drive
# drive.mount('/content/drive')
# data_dir = '/content/drive/My Drive/sirky/image_regions_64_050'

# save model weights, plotted imgs/charts
output_location = None

""" Load dataset """
data_dir = 'image_regions_64_050'

class_names, train_ds, val_ds, val_ds_batch, class_weights = get_dataset(data_dir)
num_classes = len(class_names)

""" Logging """
logs_folder = 'logs'

os.makedirs(logs_folder, exist_ok=True)

logdir = os.path.join(logs_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, profile_batch='300,400')

""" Create/Load a model """
model = models.fully_conv(num_classes, weight_init_idx=1)

# unused
# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate)

# print(model.summary())
not_printed = True

scce_base = tf.losses.SparseCategoricalCrossentropy(from_logits=False)

accu_base = tf.metrics.SparseCategoricalAccuracy()


def sca(y_true, y_pred):
    """Calculates how often predictions matches integer labels.

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    Args:
      y_true: Integer ground truth values.
      y_pred: The prediction values.

    Returns:
      Sparse categorical accuracy values.
    """
    y_pred_rank = ops.convert_to_tensor_v2(y_pred).shape.ndims
    y_true_rank = ops.convert_to_tensor_v2(y_true).shape.ndims
    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
            K.int_shape(y_true)) == len(K.int_shape(y_pred))):
        y_true = array_ops.squeeze(y_true, [-1])
    y_pred = math_ops.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast them
    # to match.
    if K.dtype(y_pred) != K.dtype(y_true):
        y_pred = math_ops.cast(y_pred, K.dtype(y_true))

    return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())


y_true_sub = None
# cnt = 0


@tf.function
def accu(y_true, y_pred):
    """
    SparseCategoricalAccuracy metric

    input reshaped from (Batch, 1, 1, 8) to (Batch, 8)
    """

    """
    # unused
    global not_printed
    if not_printed:
        print([tf.shape(y_pred)[k] for k in range(4)])
        not_printed = False
    """

    y_pred_reshaped = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[3]])

    if tf.math.equal(tf.math.reduce_max(y_pred), 0.125) is not None:
        tf.print(y_true)
        y_true = tf.constant([7])

    return accu_base(y_true, y_pred_reshaped)


saved_model_path = os.path.join('models_saved', model.name)

# Load saved model
load_module = False
epochs_trained = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if load_module:
    model = tf.keras.models.load_model(saved_model_path)
else:
    """ Train the model"""
    model.compile(
        optimizer='adam',
        loss=scce_base,
        metrics=[accu, ])  # tf.keras.metrics.SparseCategoricalAccuracy(), 'sparse_categorical_crossentropy'

    epochs = 100

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=(epochs + epochs_trained),
        initial_epoch=epochs_trained,
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(monitor='val_accu',
                                             patience=10,
                                             restore_best_weights=True)
                   ],
        class_weight=class_weights
    )
    epochs_trained += epochs

    """Save model weights"""
    if output_location:
        try:
            model.save(saved_model_path)
        except Exception as e:
            print(e)

    # false predictions + confusion map
    visualize_results(val_ds, model, output_location, class_names, epochs_trained)
    visualize_results(train_ds, model, output_location, class_names, epochs_trained)

    """Predict full image"""
    predict_full_image(model, class_names, 'heatmaps')


"""
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=10):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# unused
def scheduler(epoch, lr, start=10, decay=-0.1):
    if epoch < start:
        return lr
    else:
        return lr * tf.math.exp(decay)


lr_sched = tf.keras.callbacks.LearningRateScheduler(scheduler)
"""
