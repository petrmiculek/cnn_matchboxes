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
from itertools import product

from datasets import get_dataset
from show_results import visualize_results, predict_full_image
from src_util.labels import load_labels
import models
import util

from IPython.display import Image, display
import matplotlib.cm as cm


# extracted for timing purposes
def heatmaps_all(model, class_names, name):
    folder = 'sirky_validation'
    labels = list(load_labels(folder + os.sep + 'labels.csv', use_full_path=False))
    for file in labels:
        predict_full_image(model, class_names,
                           img_path=folder + os.sep + file,
                           heatmap_alpha=0.6,
                           # output_location='heatmaps_fixed' + name,
                           output_location=None,
                           show_figure=True)


if __name__ == '__main__':
    print(f'{tf.__version__=}')

    if len(tf.config.list_physical_devices('GPU')) == 0:
        print('no available GPUs')
        sys.exit(0)

    # from google.colab import drive
    # drive.mount('/content/drive')
    # data_dir = '/content/drive/My Drive/sirky/image_regions_64_050'

    # save model weights, plotted imgs/charts
    output_location = 'comparison'

    """ Load dataset """
    data_dir = 'image_regions_32_050'

    train_ds, class_names, class_weights = get_dataset(data_dir)
    val_ds, _, _ = get_dataset(data_dir + '_val')
    num_classes = len(class_names)

    """ Logging """
    logs_folder = 'logs'

    os.makedirs(logs_folder, exist_ok=True)

    logdir = os.path.join(logs_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, profile_batch='300,400')

    # for i, (logits, augment) in enumerate(list(product([True, False], [True, False]))):

    """ Create/Load a model """
    augment = False

    name = util.safestr(f'{augment=}')
    model = models.fully_fully_conv(num_classes, name_suffix=name, weight_init_idx=1)

    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        model = tf.keras.Sequential([data_augmentation, model, ], name=model.name)

    # unused
    # learning_rate = CustomSchedule(d_model)
    # optimizer = tf.keras.optimizers.Adam(learning_rate)

    not_printed = True

    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = util.Accu(name='accu_custom')
    lr_sched = tf.keras.callbacks.LearningRateScheduler(util.lr_scheduler)

    epochs_trained = 0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    """ Train the model"""
    model.compile(
        optimizer='adam',
        loss=scce_loss,
        metrics=[accu, ])

    epochs = 100

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        validation_freq=5,
        epochs=(epochs + epochs_trained),
        initial_epoch=epochs_trained,
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(monitor='accu_custom',  # val_accu_free_lunch
                                             patience=10,
                                             restore_best_weights=True),
            lr_sched
        ],
        class_weight=class_weights
    )
    epochs_trained += epochs

    """Evaluate model"""

    # output_location = None
    visualize_results(model, val_ds, class_names, epochs_trained, output_location, show_figure=True)
    visualize_results(model, train_ds, class_names, epochs_trained, output_location, show_figure=True)

    """Predict full image"""
    heatmaps_all(model, class_names, name)

    # sys.exit(0)

"""dump"""

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

"""
