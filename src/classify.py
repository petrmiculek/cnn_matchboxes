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

if __name__ == '__main__':
    print(f'{tf.__version__=}')

    if len(tf.config.list_physical_devices('GPU')) == 0:
        print('no available GPUs')
        sys.exit(0)

    # save model weights, plotted imgs/charts
    output_location = 'comparison'

    """ Load dataset """
    data_dir = 'image_regions_64_050'

    train_ds, class_names, class_weights = get_dataset(data_dir)
    val_ds, _, _ = get_dataset(data_dir + '_val')
    num_classes = len(class_names)

    """ Logging """
    logs_folder = 'logs'

    os.makedirs(logs_folder, exist_ok=True)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(logs_folder, time)
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, profile_batch='300,400')

    """ Create/Load a model """
    augment = True

    name = safestr('aug-{}_{}'.format(augment, time))
    base_model = models.fcn_residual_1(num_classes, name_suffix=name)
    base_model.summary()
    tf.keras.utils.plot_model(base_model, base_model.name + "_architecture_graph.png", show_shapes=True)

    if augment:
        data_augmentation = tf.keras.Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.2, fill_mode='reflect'),
            CenterCrop(32, 32)
        ])

        model = tf.keras.Sequential([data_augmentation, base_model, ], name='aug' + base_model.name)
    else:
        model = base_model

    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = util.Accu(name='accu_custom')  # ~= SparseCategoricalAccuracy
    lr_sched = tf.keras.callbacks.LearningRateScheduler(util.lr_scheduler)

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

    epochs = 1

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

    base_model.load_weights('models_saved/tff_w128_l18augTrue_20210224183906')

    # model.save_weights('models_saved' + os.sep + name)  # todo fix

    """Evaluate model"""

    # output_location = None

    visualize_results(model, val_ds, class_names, epochs_trained, output_location, show_figure=True, show_misclassified=False)
    visualize_results(model, train_ds, class_names, epochs_trained, output_location, show_figure=True)

    """Predict full image"""
    heatmaps_all(base_model, class_names, name, val=True)
    heatmaps_all(base_model, class_names, name, val=False)

    """
    # Per layer activations
    layers_intermediate_ = [layer.output for layer in crop_model.layers[2].layers]
    feat_extraction_model = tf.keras.Model(inputs=crop_model.input, outputs=layers_intermediate_)

    show_layer_activations(crop_model, feat_extraction_model, val_ds, class_names)
    """
"""dump"""
# from google.colab import drive
# drive.mount('/content/drive')
# data_dir = '/content/drive/My Drive/sirky/image_regions_64_050'


# for i, (logits, augment) in enumerate(list(product([True, False], [True, False]))):

"""
tu sluka steskne v rákosí blíž kraje
a kachna vodní s peřím zelenavým
jak duhovými barvami když hraje
se nese v dálce prachem slunce žhavý
"""
