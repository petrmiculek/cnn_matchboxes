# -*- coding: utf-8 -*-
"""
Google colab live version from ~december
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG

"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, CenterCrop
import datetime
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product
from contextlib import redirect_stdout

from datasets import get_dataset
from show_results import visualize_results, predict_full_image, show_layer_activations, heatmaps_all
from src_util.labels import load_labels
import models
import util
from src_util.general import safestr

from IPython.display import Image, display
import matplotlib.cm as cm


def get_model(model_factory, num_classes, name_suffix=''):
    base_model = model_factory(num_classes, name_suffix=name_suffix)
    base_model.summary()

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        RandomFlip("horizontal"),
        RandomRotation(1/8, fill_mode='reflect'),
        CenterCrop(32, 32),
    ], name='aug_only')

    model = tf.keras.Sequential([data_augmentation, base_model], name=base_model.name + '_full')
    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = util.Accu(name='accu')  # ~= SparseCategoricalAccuracy

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss=scce_loss,
        metrics=[accu,
                 # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name='t'),
                 # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False, name='f'),
                 ])

    # lr_sched = LearningRateScheduler(util.lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='accu', factor=0.2,
                                  patience=5, min_lr=1e-7)

    tensorboard_callback = TensorBoard(logs_dir, histogram_freq=1, profile_batch='300,400')
    callbacks = [
        tensorboard_callback,
        tf.keras.callbacks.EarlyStopping(monitor='accu',
                                         patience=10),
        # lr_sched,
        reduce_lr,
    ]

    return base_model, model, data_augmentation, callbacks


if __name__ == '__main__':

    data_dir = 'image_regions_64_050'

    time = safestr(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logs_dir = os.path.join('logs', time)

    print(f'{tf.__version__=}')
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print('no available GPUs')
        sys.exit(0)

    """ Load dataset """
    train_ds, class_names, class_weights = get_dataset(data_dir)
    val_ds, _, _ = get_dataset(data_dir + '_val')
    num_classes = len(class_names)

    """ Create/Load a model """
    base_model, model, data_augmentation, callbacks = get_model(models.fcn_residual_1, num_classes, name_suffix=time)

    """ Model outputs dir """
    output_location = os.path.join('outputs', model.name)
    if not os.path.isdir(output_location):
        os.makedirs(output_location, exist_ok=True)
    tf.keras.utils.plot_model(base_model, os.path.join(output_location, base_model.name + "_architecture.png"), show_shapes=True)

    stdout = sys.stdout
    out_stream = open(os.path.join(output_location, 'stdout.txt'), 'w')
    # print = util.print_both(os.path.join(output_location, 'stdout.txt'))
    sys.stdout = util.DuplicateStream(sys.stdout, out_stream)

    """ TensorBoard loggging """
    os.makedirs(logs_dir, exist_ok=True)
    file_writer = tf.summary.create_file_writer(logs_dir + "/metrics")
    file_writer.set_as_default()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    epochs_trained = 0
    epochs = 100

    """ Train the model"""
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        validation_freq=5,
        epochs=(epochs + epochs_trained),
        initial_epoch=epochs_trained,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2  # one line per epoch
    )
    epochs_trained += epochs

    # base_model.load_weights('models_saved/tff_w128_l18augTrue_20210224183906')
    # base_model.load_weights('models_saved/non_maxpoolaugTrue_20210228212535')

    model.save_weights(os.path.join('models_saved', model.name + '2'))

    """Evaluate model"""
    # output_location = None  # do-not-save flag
    val_accu = visualize_results(model, val_ds, class_names, epochs_trained, output_location=output_location, show=True, misclassified=True, val=True)
    visualize_results(model, train_ds, class_names, epochs_trained, output_location=output_location, show=True)

    if val_accu < 80.0:  # %
        print('Val accu too low:', val_accu)
        sys.exit(0)

    """Full image prediction"""
    heatmaps_all(base_model, class_names, val=True, output_location=output_location)
    heatmaps_all(base_model, class_names, val=False, output_location=output_location, show=True)

    """Per layer activations"""
    show_layer_activations(base_model, data_augmentation, val_ds, class_names, show=False, output_location=output_location)

"""dump"""
# from google.colab import drive
# drive.mount('/content/drive')
# data_dir = '/content/drive/My Drive/sirky/image_regions_64_050'


# for i, (logits, augment) in enumerate(list(product([True, False], [True, False]))):

"""
tu sluka steskne v rákosí blíž kraje
a kachna vodní s peřím zelenavým
jak duhovými barvami když hraje
se nese v dálce prachem slunce žhavým
"""
