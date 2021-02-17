# -*- coding: utf-8 -*-
"""
Google colab live version from ~december
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG



"""
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import datetime
import os

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


def heatmaps_all(model, class_names, name, val=True):
    folder = 'sirky' + '_validation' * val
    labels = list(load_labels(folder + os.sep + 'labels.csv', use_full_path=False))
    # if not val:
    #     labels = labels[-7:-5]

    for file in labels:  # [-7:-5] for train_ds
        predict_full_image(model, class_names,
                           img_path=folder + os.sep + file,
                           heatmap_alpha=0.6,
                           output_location='heatmaps_before_manual' + name,
                           # output_location=None,
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
    augment = True

    name = util.safestr('aug_{}'.format(augment))
    model = models.fully_conv_tff(num_classes, name_suffix=name)
    print(model.summary())
    tf.keras.utils.plot_model(model, model.name + "_architecture_graph.png", show_shapes=True)

    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='reflect'),
        ])

        model = tf.keras.Sequential([data_augmentation, model, ], name=model.name)

    not_printed = True

    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = util.Accu(name='accu_custom')
    lr_sched = tf.keras.callbacks.LearningRateScheduler(util.lr_scheduler)

    epochs_trained = 0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    """ Train the model"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=scce_loss,
        metrics=[accu, ])

    epochs = 40

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        validation_freq=5,
        epochs=(epochs + epochs_trained),
        initial_epoch=epochs_trained,
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(monitor='accu_custom',
                                             patience=10,
                                             ),
            lr_sched
        ],
        class_weight=class_weights
    )
    epochs_trained += epochs

    """Evaluate model"""

    # output_location = None
    visualize_results(model, val_ds, class_names, epochs_trained, output_location, show_figure=True, show_misclassified=False)
    visualize_results(model, train_ds, class_names, epochs_trained, output_location, show_figure=True)

    """Predict full image"""
    heatmaps_all(model, class_names, name, val=True)
    heatmaps_all(model, class_names, name, val=False)

    # sys.exit(0)

"""dump"""
