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

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product

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
    tf.keras.utils.plot_model(base_model, base_model.name + "_architecture.png", show_shapes=True)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        RandomFlip("horizontal"),
        RandomRotation(1/8, fill_mode='reflect'),
        CenterCrop(32, 32),
    ], name='aug_only')

    model = tf.keras.Sequential([data_augmentation, base_model], name=base_model.name + '_full')
    return base_model, model, data_augmentation


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
    base_model, model, data_augmentation = get_model(models.fcn_residual_1, num_classes, name_suffix=time)
    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = util.Accu(name='accu_custom')  # ~= SparseCategoricalAccuracy
    lr_sched = tf.keras.callbacks.LearningRateScheduler(util.lr_scheduler)

    """ Logging (unrelated to currdir) """
    os.makedirs(logs_dir, exist_ok=True)
    file_writer = tf.summary.create_file_writer(logs_dir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logs_dir, histogram_freq=1, profile_batch='300,400')
    epochs_trained = 0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    """ Train the model"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=scce_loss,
        metrics=[accu,
                 # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name='t'),
                 # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False, name='f'),
                 ])

    epochs = 50

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        validation_freq=5,
        epochs=(epochs + epochs_trained),
        initial_epoch=epochs_trained,
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(monitor='accu_custom',
                                             patience=10),
            lr_sched
        ],
        class_weight=class_weights
    )
    epochs_trained += epochs

    # base_model.load_weights('models_saved/tff_w128_l18augTrue_20210224183906')
    # base_model.load_weights('models_saved/non_maxpoolaugTrue_20210228212535')

    # model.save_weights(os.path.join('models_saved', model.name))

    """ Model outputs dir """
    output_location = os.path.join('outputs', model.name)
    if not os.path.isdir(output_location):
        os.makedirs(output_location, exist_ok=True)

    """Evaluate model"""
    # output_location = None  # do-not-save flag
    visualize_results(model, val_ds, class_names, epochs_trained, output_location=output_location, show=True, show_misclassified=False, val=True)
    visualize_results(model, train_ds, class_names, epochs_trained, output_location=output_location, show=True)

    """Full image prediction"""
    heatmaps_all(base_model, class_names, val=True, output_location=output_location)
    heatmaps_all(base_model, class_names, val=False, output_location=output_location, show=False)

    """Per layer activations"""
    show_layer_activations(base_model, data_augmentation, val_ds, class_names, show=True, output_location=output_location)

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
