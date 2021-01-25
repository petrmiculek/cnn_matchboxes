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
from src_util.labels import load_labels
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
data_dir = 'image_regions_32_050'

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
model = models.fully_fully_conv(num_classes, weight_init_idx=1)

# unused
# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate)

not_printed = True

scce_base = tf.losses.SparseCategoricalCrossentropy(from_logits=False)

accu_base = tf.metrics.SparseCategoricalAccuracy()


@tf.function
def accu(y_true, y_pred):
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

    return accu_base(y_avoid_free, y_pred_reshaped)


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
    visualize_results(model, val_ds, class_names, epochs_trained, output_location)
    visualize_results(model, train_ds, class_names, epochs_trained, output_location)

    """Predict full image"""
    labels = list(load_labels('sirky/labels.csv', use_full_path=False))

    for file in labels:  # [-2:-1]
        predict_full_image(model, class_names,
                           img_path='sirky' + os.sep + file,
                           output_location='full_res_heatmaps',
                           show_figure=False)

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
