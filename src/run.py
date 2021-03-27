# -*- coding: utf-8 -*-
"""
Google colab live version from ~december
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG

"""

# disable profiling-related-errors
# def profile(x):
#     return x


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
from tensorboard.plugins.hparams import api as hp

# hacky, just for profiling
sys.path.extend(['/home/petrmiculek/Code/light_matches',
                 '/home/petrmiculek/Code/light_matches/src',
                 '/home/petrmiculek/Code/light_matches/src_util'])
# local files
import models
import model_ops
import util
from datasets import get_dataset
from show_results import visualize_results, predict_full_image, show_layer_activations, heatmaps_all
# from src_util.labels import load_labels_dict
from src_util.general import safestr, DuplicateStream
from logging_results import log_model_info
from mining import mine_hard_cases
from src_util.general import timing


# when profiling disabled
# def profile(x):
#     return x

# @pro file
def run(model_builder, model_kwargs={}, use_small_ds=True, augment=False, train=True, ds_dim=64):
    """
    # execute if running in Py console
    model_builder = models.dilated_64x_exp2
    model_kwargs={'pool': 'max'}
    use_small_ds=False
    augment=False
    train = True
    show = True
    ds_dim=64
    """
    # for batch-running
    # if True:
    try:
        hard_mining = False
        show = False
        scale = 0.5
        use_weights = model_kwargs['use_weights'] if 'use_weights' in model_kwargs else False

        # per image background samples
        if use_small_ds:
            bg_samples = 100
        else:
            bg_samples = 500

        data_dir = f'/data/datasets/{ds_dim}x_{int(100 * scale):03d}s_{bg_samples}bg'
        # data_dir = f'datasets/{ds_dim}x_{int(100 * scale):03d}s_{bg_samples}bg'
        checkpoint_path = util.get_checkpoint_path()

        time = safestr(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        """ Load dataset """
        val_ds, _, _ = get_dataset(data_dir + '_val')
        train_ds, class_names, class_weights = get_dataset(data_dir)
        num_classes = len(class_names)

        """ Create/Load a model """
        if train:
            base_model, model, data_augmentation, callbacks = model_ops.build_new_model(model_builder,
                                                                                        model_kwargs,
                                                                                        num_classes,
                                                                                        augment=augment,
                                                                                        name_suffix=time,
                                                                                        ds_dim=ds_dim,
                                                                                        checkpoint_path=checkpoint_path,
                                                                                        bg_samples=bg_samples,
                                                                                        )

        else:
            # load_model_name = 'dilated_64x_exp2_2021-03-19-02-11-52_full'  # trained on 500bg
            # load_model_name = 'dilated_64x_exp2_2021-03-26-04-57-57_full'  # /data/datasets/64x_050s_500bg
            load_model_name = 'dilated_64x_exp2_2021-03-27-06-36-50_full'  # /data/datasets/128x_050s_500bg
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
        print(model_kwargs)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        epochs_trained = 0

        if train:
            epochs = 100
            if hard_mining:
                epochs_to_reshuffle = 10
                curr_ds = train_ds
                while epochs_trained < epochs_total:
                    """ Train the model"""
                    model.fit(
                        curr_ds,
                        validation_data=val_ds,
                        validation_freq=5,
                        epochs=(epochs_to_reshuffle + epochs_trained),
                        initial_epoch=epochs_trained,
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=2  # one line per epoch
                    )

                    hard_ds = timing(mine_hard_cases)(base_model, train_ds)

                    # expand for keypoints and background
                    curr_ds = tf.data.experimental.sample_from_datasets([train_ds, hard_ds], weights=[0.5, 0.5])
                    curr_ds = curr_ds.batch(64).prefetch(2)

                    epochs_trained += epochs
            else:
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    validation_freq=5,
                    epochs=(epochs + epochs_trained),
                    initial_epoch=epochs_trained,
                    callbacks=callbacks,
                    # class_weight=class_weights,  # WITHOUT
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

        # show = True
        # output_location = None  # do-not-save flag
        val_accu = visualize_results(model, val_ds, class_names, epochs_trained, output_location=output_location,
                                     show=show, misclassified=True, val=True)
        visualize_results(model, train_ds, class_names, epochs_trained, output_location=output_location, show=show)

        if val_accu < 90.0:  # %
            print('Val accu too low:', val_accu, 'skipping heatmaps')
            return
            # sys.exit(0)

        """Full image prediction"""
        heatmaps_all(base_model, class_names, val=True, output_location=output_location, show=show,
                     epochs_trained=epochs_trained)
        heatmaps_all(base_model, class_names, val=False, output_location=output_location, show=show,
                     epochs_trained=epochs_trained)

        """Per layer activations"""
        show_layer_activations(base_model, data_augmentation, val_ds, class_names, show=show,
                               output_location=output_location)

        # restore original stdout
        sys.stdout = stdout_orig
    except Exception as ex:
        print(ex, file=sys.stderr)


def tf_init():
    print(f'{tf.__version__=}')
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) == 0:
        print('no GPU available')
        sys.exit(0)
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # debugging
    # tf.executing_eagerly()
    # tf.config.experimental_functions_run_eagerly()
    # tf.config.experimental_run_functions_eagerly(True)
    # tf.config.experimental_functions_run_eagerly()
    # tf.executing_eagerly()


if __name__ == '__main__':
    tf_init()

    run(models.dilated_64x_exp2, use_small_ds=True, train=True, model_kwargs={'pool': 'max'})
    # run(models.dilated_64x_exp2, use_small_ds=False, train=False)
    pass
