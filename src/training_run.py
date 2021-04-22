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
from models import *
import model_build
from datasets import get_dataset
from eval_images import full_prediction_all
from eval_samples import evaluate_model, show_layer_activations
from logs import log_model_info
from mining import mine_hard_cases
from src_util.general import safestr, DuplicateStream, timing, get_checkpoint_path
import config


def run(model_builder, hparams):
    """Perform a single training run
    """
    if False:
        # provided by caller
        config.epochs = 50
        config.dataset_dim = 64
        config.dataset_size = 200
        config.augment = True
        config.train = True
        config.show = False
        config.use_weights = False
        config.scale = 0.5
        config.center_crop_fraction = 0.5
        config.batch_size = 128

    hard_mining = False

    # for batch-running
    # if True:
    try:
        dataset_dir = f'/data/datasets/{config.dataset_dim}x_{int(100 * config.scale):03d}s_{config.dataset_size}bg'
        print('Loading dataset from:', dataset_dir)
        config.checkpoint_path = get_checkpoint_path()

        time = safestr(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        """ Load dataset """
        use_weights = hparams['class_weights'] if 'class_weights' in hparams else None
        val_ds, _, _ = get_dataset(dataset_dir + '_val', batch_size=config.batch_size)
        train_ds, config.class_names, class_weights = \
            get_dataset(dataset_dir, weights=use_weights, batch_size=config.batch_size)

        """ Create/Load a model """
        if config.train:
            base_model, model, aug_model = model_build.build_new_model(model_builder, hparams, name_suffix=time)
            callbacks = model_build.get_callbacks()

        else:
            load_model_name = 'dilated_64x_exp2_2021-03-29-15-58-47_full'  # /data/datasets/128x_050s_1000bg
            model_config_path = os.path.join('outputs', load_model_name, 'model_config.json')
            weights_path = os.path.join('models_saved', load_model_name)

            base_model, model, aug_model = model_build.load_model(model_config_path, weights_path)
            callbacks = model_build.get_callbacks()
            config.epochs_trained = 123

        """ Model outputs dir """
        config.output_location = os.path.join('outputs', model.name + '_reloaded' * (not config.train))
        if not os.path.isdir(config.output_location):
            os.makedirs(config.output_location, exist_ok=True)

        """ Copy stdout to file """
        stdout_orig = sys.stdout
        out_stream = open(os.path.join(config.output_location, 'stdout.txt'), 'a')
        sys.stdout = DuplicateStream(sys.stdout, out_stream)

        log_model_info(model, config.output_location)
        print({h: hparams[h] for h in hparams})

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.get_logger().setLevel('ERROR')  # suppress warnings about early-stopping and model-checkpoints

        config.epochs_trained = 0

        if config.train:
            if hard_mining:

                epochs_to_reshuffle = 10
                curr_ds = train_ds
                while config.epochs_trained < config.epochs:
                    """ Train the model"""
                    model.fit(
                        curr_ds,
                        validation_data=val_ds,
                        validation_freq=5,
                        epochs=(epochs_to_reshuffle + config.epochs_trained),
                        initial_epoch=config.epochs_trained,
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=2  # one line per epoch
                    )

                    hard_ds = timing(mine_hard_cases)(base_model, train_ds)

                    # expand for keypoints and background
                    curr_ds = tf.data.experimental.sample_from_datasets([train_ds, hard_ds], weights=[0.5, 0.5])
                    curr_ds = curr_ds.batch(64).prefetch(2)

                    config.epochs_trained += config.epochs
            else:
                try:
                    model.fit(
                        train_ds,
                        validation_data=val_ds,
                        validation_freq=5,
                        epochs=(config.epochs + config.epochs_trained),
                        initial_epoch=config.epochs_trained,
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=2  # one line per epoch
                    )
                except KeyboardInterrupt:
                    print('Training stopped preemptively')
                config.epochs_trained += config.epochs
            try:
                if config.epochs_trained > 5:
                    model.load_weights(config.checkpoint_path)  # checkpoint
            except NotFoundError:
                # might not be present if trained for <K epochs
                pass

            base_model.save_weights(os.path.join('models_saved', model.name))
            # base_model.load_weights('models_saved/residual_20210305142236_full')

        """Evaluate model"""

        # config.show = True
        # config.output_location = None  # do-not-save flag
        val_accu = evaluate_model(model, val_ds, val=True, output_location=config.output_location,
                                  show=config.show, misclassified=False)

        evaluate_model(model, train_ds, val=False, output_location=config.output_location, show=config.show)

        if val_accu < 95.0:  # %
            print('Val accu too low:', val_accu, 'skipping heatmaps')
            return
            # sys.exit(0)

        """Full image prediction"""
        pix_mse_val, dist_mse_val, count_mae_val = \
            full_prediction_all(base_model, val=True, output_location=config.output_location, show=config.show)
        pix_mse_train, dist_mse_train, count_mae_train = \
            full_prediction_all(base_model, val=False, output_location=config.output_location, show=False)

        val_metrics = model.evaluate(val_ds, verbose=0)  # 5 metrics, as per model_ops.compile_model
        pr_value_val = val_metrics[4]

        print('mse: {}'.format(dist_mse_train))
        print('mse_val: {}'.format(dist_mse_val))
        print('pr_value_val: {}'.format(pr_value_val))

        if config.train:
            with tf.summary.create_file_writer(config.run_logs_dir + '/hparams').as_default():
                hp.hparams(hparams, trial_id=config.model_name)
                tf.summary.scalar('pix_mse_val', pix_mse_val, step=config.epochs_trained)
                tf.summary.scalar('dist_mse_val', dist_mse_val, step=config.epochs_trained)
                tf.summary.scalar('count_mae_val', count_mae_val, step=config.epochs_trained)

                tf.summary.scalar('pix_mse_train', pix_mse_train, step=config.epochs_trained)
                tf.summary.scalar('dist_mse_train', dist_mse_train, step=config.epochs_trained)
                tf.summary.scalar('count_mae_train', count_mae_train, step=config.epochs_trained)

                tf.summary.scalar('pr_value_val', pr_value_val, step=config.epochs_trained)

        """Per layer activations"""
        show_layer_activations(base_model, aug_model, val_ds, show=False,
                               output_location=config.output_location)

        # restore original stdout
        sys.stdout = stdout_orig

    except Exception as ex:
        print(type(ex), ex, '\n\n', file=sys.stderr)
        # raise


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

    config.dataset_size = 1000
    config.train = True
    # train dim decided by model
    config.dataset_dim = 128
    config.augment = 3
    config.use_weights = False
    config.show = False
    config.scale = 0.25
    config.center_crop_fraction = 0.5
    config.epochs = 50

    # model params
    hp_base_width = hp.HParam('base_width', hp.Discrete([8, 16, 32]))
    # non-model params
    hp_aug_level = hp.HParam('aug_level', hp.Discrete([0, 1, 2, 3]))
    hp_class_weights = hp.HParam('ds_bg_samples', hp.Discrete(['none', 'inverse_frequency', 'effective_number']))
    hp_crop_fraction = hp.HParam('crop_fraction', hp.Discrete([0.5, 1.0]))
    hp_ds_bg_samples = hp.HParam('ds_bg_samples', hp.Discrete([200, 700, 1000]))
    hp_scale = hp.HParam('scale', hp.Discrete([0.25, 0.5]))

    with tf.summary.create_file_writer('logs').as_default():
        hp.hparams_config(
            hparams=[
                hp_aug_level,
                hp_base_width,
                hp_class_weights,
                hp_crop_fraction,
                hp_ds_bg_samples,
                hp_scale,
            ],
            metrics=[
                hp.Metric('pix_mse_train'),
                hp.Metric('dist_mse_train'),
                hp.Metric('count_mae_train'),

                hp.Metric('pix_mse_val'),
                hp.Metric('dist_mse_val'),
                hp.Metric('count_mae_val'),

                hp.Metric('pr_value_val')]
        )

    m = parameterized(recipe_51x_odd)
    hparams = {
        'base_width': 32,

        'aug_level': config.augment,
        'ds_bg_samples': config.dataset_size,
        'scale': config.scale,
        'crop_fraction': config.center_crop_fraction,
        # 'tail_downscale': 2  # try if logging to tb fails
    }
    run(m, hparams)

    pass
