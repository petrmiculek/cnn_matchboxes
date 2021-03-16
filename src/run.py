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
from tensorflow.python.framework.errors_impl import NotFoundError

import cv2 as cv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Image, display

# local files
import models
import model_ops
import util
from datasets import get_dataset
from show_results import visualize_results, predict_full_image, show_layer_activations, heatmaps_all
from src_util.labels import load_labels
from src_util.general import safestr, DuplicateStream
from logging_results import log_model_info


def run(model_builder, model_kwargs={}, use_small_ds=True, augment=False, train=True):
    """
    # execute if running in Py console
    model_builder = models.dilated_32x_exp2
    model_kwargs={}
    use_small_ds=True
    augment=False
    train = True
    show = True

    """
    # for batch-running
    show = False
    try:
        dim = 64  # training sample dim - not really
        scale = 0.5

        # per image background samples
        if use_small_ds:
            bg_samples = 100
        else:
            bg_samples = 500

        data_dir = f'image_regions_{dim}_{int(100 * scale):03d}_bg{bg_samples}'
        checkpoint_path = util.get_checkpoint_path()

        time = safestr(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        print(f'{tf.__version__=}')
        if len(tf.config.list_physical_devices('GPU')) == 0:
            print('no GPU available')
            sys.exit(0)

        """ Load dataset """
        train_ds, class_names, class_weights = get_dataset(data_dir)
        val_ds, _, _ = get_dataset(data_dir + '_val')
        num_classes = len(class_names)

        # tf.executing_eagerly()
        # tf.config.experimental_functions_run_eagerly()
        # tf.config.experimental_run_functions_eagerly(True)
        # tf.config.experimental_functions_run_eagerly()
        # tf.executing_eagerly()

        """ Create/Load a model """
        if train:
            base_model, model, data_augmentation, callbacks = model_ops.build_new_model(model_builder,
                                                                                        model_kwargs,
                                                                                        num_classes,
                                                                                        augment=augment,
                                                                                        name_suffix=time,
                                                                                        checkpoint_path=checkpoint_path,
                                                                                        bg_samples=bg_samples)
        else:
            load_model_name = 'dilated20210309161954_full'  # trained on 500bg
            config = os.path.join('outputs', load_model_name, 'model_config.json')
            weights = os.path.join('models_saved', load_model_name)

            base_model, model, data_augmentation, callbacks = model_ops.load_model(config, weights)

        """ Model outputs dir """

        output_location = os.path.join('outputs', model.name + '_reloaded' * (not train))
        if not os.path.isdir(output_location):
            os.makedirs(output_location, exist_ok=True)

        stdout_orig = sys.stdout
        out_stream = open(os.path.join(output_location, 'stdout.txt'), 'a')
        sys.stdout = DuplicateStream(sys.stdout, out_stream)

        log_model_info(model, output_location)

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

            try:
                if epochs_trained > 5:
                    model.load_weights(checkpoint_path)  # checkpoint
            except NotFoundError:
                # might not be present if trained for <K epochs
                pass

            base_model.save_weights(os.path.join('models_saved', model.name))
            # base_model.load_weights('models_saved/residual_20210305142236_full')

        """Evaluate model"""

        # output_lo cation = None  # do-not-save flag
        val_accu = visualize_results(model, val_ds, class_names, epochs_trained, output_location=output_location,
                                     show=show, misclassified=True, val=True)
        visualize_results(model, train_ds, class_names, epochs_trained, output_location=output_location, show=show)

        if val_accu < 80.0:  # %
            print('Val accu too low:', val_accu)
            # sys.exit(0)

        """Full image prediction"""
        heatmaps_all(base_model, class_names, val=True, output_location=output_location, show=show)
        heatmaps_all(base_model, class_names, val=False, output_location=output_location, show=show)

        """Per layer activations"""
        show_layer_activations(base_model, data_augmentation, val_ds, class_names, show=show,
                               output_location=output_location)

        # restore original stdout
        sys.stdout = stdout_orig
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    # run(models.fcn_residual_32x_18l, use_small_ds=True, augment=False)
    pass
