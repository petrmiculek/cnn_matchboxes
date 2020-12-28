# -*- coding: utf-8 -*-
"""
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG

todo:
fix annotation sirky (1 img)

21. 12.
zkus znovu projit tutorial na fine-tuning, pripadne grad cam



"""
import os
import sys

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

print(tf.config.list_physical_devices('GPU'))  # show available GPUs

# from google.colab import drive
# drive.mount('/content/drive')
# data_dir = '/content/drive/My Drive/sirky/image_regions_64_050'

# save model weights, plotted imgs/charts
save_outputs = False

""" Load dataset """
data_dir = 'image_regions_64_050'

class_names, train_ds, val_ds, val_ds_batch, class_weights = get_dataset(data_dir)
num_classes = len(class_names)

""" Logging """
logs_folder = 'logs'

os.makedirs(logs_folder, exist_ok=True)

logdir = os.path.join(logs_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

""" Create/Load a model """
model = models.fully_conv(num_classes)

print(model.summary())

# sys.exit(0)

saved_model_path = os.path.join('models_saved', model.name)

# Load saved model
load_module = False

if load_module:
    model = tf.keras.models.load_model(saved_model_path)
    epochs_trained = 0
else:
    """ Train the model"""
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', 'sparse_categorical_crossentropy'])

    epochs = 20

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             restore_best_weights=True)
                   ],
        class_weight=class_weights
    )

    epochs_trained = len(history.epoch)

    """Save model weights"""
    if save_outputs:
        model.save(saved_model_path)

# false predictions + confusion map
visualize_results(val_ds, model, save_outputs, class_names, epochs_trained)
visualize_results(train_ds, model, save_outputs, class_names, epochs_trained)

"""Predict full image"""
predict_full_image(model, class_names)

"""
# https://keras.io/guides/transfer_learning/
resnet = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    # weights=None,
    input_shape=(64, 64, 3),
    pooling='avg',  # average pooling into single prediction
    classes=2)  # does not seem to have any meaning
resnet.trainable = False

inputs = tf.keras.Input(shape=(64, 64, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = resnet(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
# x = tf.keras.layers.Flatten()(x)
# outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
# outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs, x)
"""
