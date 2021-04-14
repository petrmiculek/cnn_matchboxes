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

# external libs
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorboard.plugins.hparams import api as hp

# import cv2 as cv
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from IPython.display import Image, display

# hacky, just for profiling
sys.path.extend(['/home/petrmiculek/Code/light_matches',
                 '/home/petrmiculek/Code/light_matches/src',
                 '/home/petrmiculek/Code/light_matches/src_util'])

# local files
import models
import model_ops
import util
from datasets import get_dataset
from eval_images import full_prediction_all
from eval_samples import evaluate_model, show_layer_activations
from logging_results import log_model_info
from mining import mine_hard_cases
from src_util.general import safestr, DuplicateStream, timing
import run_config


def run(model_builder, hparams):
    """Perform a single training run
    """
    if False:
        # provided by caller
        run_config.dataset_dim = 64
        run_config.dataset_size = 200
        run_config.augment = True
        run_config.train = True
        run_config.show = False
        run_config.use_weights = False
        run_config.scale = 0.5
        run_config.center_crop_fraction = 0.5

    hard_mining = False

    # for batch-running
    # try:
    if True:
        dataset_dir = f'/data/datasets/{run_config.dataset_dim}x_{int(100 * run_config.scale):03d}s_{run_config.dataset_size}bg'
        run_config.checkpoint_path = util.get_checkpoint_path()

        time = safestr(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        """ Load dataset """
        val_ds, _, _ = get_dataset(dataset_dir + '_val')
        train_ds, run_config.class_names, class_weights = get_dataset(dataset_dir, use_weights=run_config.use_weights)

        """ Create/Load a model """
        if run_config.train:
            base_model, model, aug_model = model_ops.build_new_model(model_builder, hparams, name_suffix=time)
            callbacks = model_ops.get_callbacks()

        else:
            load_model_name = 'dilated_64x_exp2_2021-03-29-15-58-47_full'  # /data/datasets/128x_050s_1000bg
            model_config_path = os.path.join('outputs', load_model_name, 'model_config.json')
            weights_path = os.path.join('models_saved', load_model_name)

            base_model, model, aug_model = model_ops.load_model(model_config_path, weights_path)
            callbacks = model_ops.get_callbacks()
            run_config.epochs_trained = 123

        """ Model outputs dir """
        run_config.output_location = os.path.join('outputs', model.name + '_reloaded' * (not run_config.train))
        if not os.path.isdir(run_config.output_location):
            os.makedirs(run_config.output_location, exist_ok=True)

        """ Copy stdout to file """
        stdout_orig = sys.stdout
        out_stream = open(os.path.join(run_config.output_location, 'stdout.txt'), 'a')
        sys.stdout = DuplicateStream(sys.stdout, out_stream)

        log_model_info(model, run_config.output_location)
        print({h: hparams[h] for h in hparams})

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.get_logger().setLevel('ERROR')  # suppress warnings about early-stopping and model-checkpoints

        run_config.epochs_trained = 0

        if run_config.train:
            epochs = 50
            if hard_mining:

                epochs_to_reshuffle = 10
                curr_ds = train_ds
                while run_config.epochs_trained < epochs:
                    """ Train the model"""
                    model.fit(
                        curr_ds,
                        validation_data=val_ds,
                        validation_freq=5,
                        epochs=(epochs_to_reshuffle + run_config.epochs_trained),
                        initial_epoch=run_config.epochs_trained,
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=2  # one line per epoch
                    )

                    hard_ds = timing(mine_hard_cases)(base_model, train_ds)

                    # expand for keypoints and background
                    curr_ds = tf.data.experimental.sample_from_datasets([train_ds, hard_ds], weights=[0.5, 0.5])
                    curr_ds = curr_ds.batch(64).prefetch(2)

                    run_config.epochs_trained += epochs
            else:
                try:
                    model.fit(
                        train_ds,
                        validation_data=val_ds,
                        validation_freq=5,
                        epochs=(epochs + run_config.epochs_trained),
                        initial_epoch=run_config.epochs_trained,
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=2  # one line per epoch
                    )
                except KeyboardInterrupt:
                    print('Training stopped preemptively')
                run_config.epochs_trained += epochs
            try:
                if run_config.epochs_trained > 5:
                    model.load_weights(run_config.checkpoint_path)  # checkpoint
            except NotFoundError:
                # might not be present if trained for <K epochs
                pass

            base_model.save_weights(os.path.join('models_saved', model.name))
            # base_model.load_weights('models_saved/residual_20210305142236_full')

        """Evaluate model"""

        # run_config.show = True
        # run_config.output_location = None  # do-not-save flag
        val_accu = evaluate_model(model, val_ds, val=True, output_location=run_config.output_location,
                                  show=run_config.show, misclassified=False)

        evaluate_model(model, train_ds, val=False, output_location=run_config.output_location, show=run_config.show)

        if val_accu < 95.0:  # %
            print('Val accu too low:', val_accu, 'skipping heatmaps')
            return
            # sys.exit(0)

        """Full image prediction"""
        avg_mse_val = full_prediction_all(base_model, val=True, output_location=run_config.output_location,
                                          show=run_config.show)
        avg_mse_train = full_prediction_all(base_model, val=False, output_location=run_config.output_location,
                                            show=False)

        val_metrics = model.evaluate(val_ds, verbose=0)  # 5 metrics, as per model_ops.compile_model
        pr_value_val = val_metrics[4]

        print('avg_mse: {}'.format(avg_mse_train))
        print('avg_mse_val: {}'.format(avg_mse_val))
        print('pr_value_val: {}'.format(pr_value_val))

        if run_config.train:
            with tf.summary.create_file_writer(run_config.run_logs_dir + '/hparams').as_default():
                hp.hparams(hparams, trial_id=run_config.model_name)
                tf.summary.scalar('mse', avg_mse_train, step=run_config.epochs_trained)
                tf.summary.scalar('mse_val', avg_mse_val, step=run_config.epochs_trained)
                tf.summary.scalar('pr_value_val', pr_value_val, step=run_config.epochs_trained)

        """Per layer activations"""
        show_layer_activations(base_model, aug_model, val_ds, show=False,
                               output_location=run_config.output_location)

        # restore original stdout
        sys.stdout = stdout_orig
    # except Exception as ex:
    #     print(ex, file=sys.stderr)


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

    run_config.dataset_size = 1000
    run_config.train = True
    # train dim decided by model
    run_config.dataset_dim = 64
    run_config.augment = True
    run_config.use_weights = False
    run_config.show = False
    run_config.scale = 0.25
    run_config.center_crop_fraction = 0.5

    # model params
    hp_base_width = hp.HParam('base_width', hp.Discrete([8, 16, 32]))

    # non-model params
    hp_ds_bg_samples = hp.HParam('ds_bg_samples', hp.Discrete([200, 700, 1000]))
    hp_augmentation = hp.HParam('augmentation', hp.Discrete([False, True]))
    hp_scale = hp.HParam('scale', hp.Discrete([0.25, 0.5]))
    hp_crop_fraction = hp.HParam('crop_fraction', hp.Discrete([0.5, 1.0]))

    with tf.summary.create_file_writer('logs').as_default():
        hp.hparams_config(
            hparams=[
                hp_base_width,
                hp_ds_bg_samples,
                hp_augmentation,
                hp_scale,
                hp_crop_fraction,
            ],
            metrics=[hp.Metric('mse')],
        )

    m = models.dilated_32x_odd
    hparams = {
        'base_width': 32,

        'augmentation': run_config.augment,
        'ds_bg_samples': run_config.dataset_size,
        'scale': run_config.scale,
        'crop_fraction': run_config.center_crop_fraction,
        # 'tail_downscale': 2  # try if logging to tb fails
    }
    run(m, hparams)

    pass
