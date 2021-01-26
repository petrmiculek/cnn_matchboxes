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
import util

from IPython.display import Image, display
import matplotlib.cm as cm

print(f'{tf.__version__=}')

if len(tf.config.list_physical_devices('GPU')) == 0:
    print('no available GPUs')
    sys.exit(0)

# plt.figure(figsize=(10, 10))
# for i in range(9):
#     augmented_image = util.Augmentation()(image)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_image[0])
#     plt.axis("off")

# from google.colab import drive
# drive.mount('/content/drive')
# data_dir = '/content/drive/My Drive/sirky/image_regions_64_050'

# save model weights, plotted imgs/charts
output_location = None

""" Load dataset """
data_dir = 'image_regions_32_050'

class_names, train_ds, val_ds, val_ds_batch, class_weights = get_dataset(data_dir, augmentation=True)
num_classes = len(class_names)

""" Logging """
logs_folder = 'logs'

os.makedirs(logs_folder, exist_ok=True)

logdir = os.path.join(logs_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, profile_batch='300,400')

""" Create/Load a model """
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

model = tf.keras.Sequential([data_augmentation,
                             models.fully_fully_conv(num_classes, weight_init_idx=1),
                             ])

# unused
# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate)

not_printed = True

scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
accu = util.Accu()

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
        loss=scce_loss,
        metrics=[accu, ])  # tf.keras.metrics.SparseCategoricalAccuracy(), 'sparse_categorical_crossentropy'

    epochs = 1

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
    output_location = None
    visualize_results(model, val_ds, class_names, epochs_trained, output_location, show_figure=True)
    visualize_results(model, train_ds, class_names, epochs_trained, output_location, show_figure=True)

    """Predict full image"""
    labels = list(load_labels('sirky/labels.csv', use_full_path=False))

    for file in labels[-2:-1]:
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
