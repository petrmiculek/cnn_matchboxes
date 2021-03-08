# -*- coding: utf-8 -*-
"""
Google colab live version from ~december
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG

"""

# stdlib
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
from itertools import product
from contextlib import redirect_stdout

# external libs
import tensorflow as tf

import cv2 as cv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Image, display

# local files
import models
import util
from datasets import get_dataset
from show_results import visualize_results, predict_full_image, show_layer_activations, heatmaps_all
from src_util.labels import load_labels
from src_util.general import safestr, DuplicateStream
from logging_results import log_model_info

if __name__ == '__main__':
    train = True
    training_sample_dim = 64
    data_dir = 'image_regions_{}_050'.format(training_sample_dim)
    checkpoint_path = '/tmp/checkpoint'

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
    if train:
        base_model, model, data_augmentation, callbacks = models.build_model(models.fcn_residual_1, num_classes,
                                                                             name_suffix=time, logs_dir=logs_dir,
                                                                             augment=False, checkpoint_path=checkpoint_path)
    else:
        models.load_model()

    """ Model outputs dir """
    output_location = os.path.join('outputs', model.name + '_re-loaded' * (not train))
    if not os.path.isdir(output_location):
        os.makedirs(output_location, exist_ok=True)
    log_model_info(model, output_location)

    stdout = sys.stdout
    out_stream = open(os.path.join(output_location, 'stdout.txt'), 'w')
    sys.stdout = DuplicateStream(sys.stdout, out_stream)

    """ TensorBoard loggging """
    os.makedirs(logs_dir, exist_ok=True)
    file_writer = tf.summary.create_file_writer(logs_dir + "/metrics")
    file_writer.set_as_default()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    epochs_trained = 0

    if train:
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

        model.load_weights(checkpoint_path)  # checkpoint
        base_model.save_weights(os.path.join('models_saved', model.name))

        # base_model.load_weights('models_saved/residual_20210305142236_full')
        # raise ValueError()

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

# for i, (logits, augment) in enumerate(list(product([True, False], [True, False]))):

"""
tu sluka steskne v rákosí blíž kraje
a kachna vodní s peřím zelenavým
jak duhovými barvami když hraje
se nese v dálce prachem slunce žhavým
"""
