# stdlib
import os
import sys

# external
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

curr_path = os.getcwd()
sys.path.extend([curr_path] + [d for d in os.listdir() if os.path.isdir(d)])

# local
from training_run import run
from model_util import tensorboard_hparams_init, tf_init
from models import *
from models_old import *
import config

# noinspection DuplicatedCode
if __name__ == '__main__':
    tf_init()
    """ model runs """

    config.dataset_size = 1000
    config.train = True
    # train dim decided by model
    config.dataset_dim = 128
    config.augment = 2
    config.show = False
    config.scale = 0.25
    config.center_crop_fraction = 0.5

    # model params
    hp_base_width = hp.HParam('base_width', hp.Discrete([16, 32]))

    # non-model params
    hp_class_weights = hp.HParam('class_weights', hp.Discrete(['none', 'inv_freq', 'eff_num']))
    hp_ds_bg_samples = hp.HParam('ds_bg_samples', hp.Discrete([200, 700, 1000]))
    hp_aug_level = hp.HParam('aug_level', hp.Discrete([0, 1, 2, 3]))
    hp_scale = hp.HParam('scale', hp.Discrete([0.25, 0.5]))
    hp_crop_fraction = hp.HParam('crop_fraction', hp.Discrete([0.5, 1.0]))
    hparams = [
        hp_aug_level,
        hp_base_width,
        hp_class_weights,
        hp_crop_fraction,
        hp_ds_bg_samples,
        hp_scale,
    ]
    tensorboard_hparams_init(hparams)

    config.datasets_root = 'datasets'

    config.batch_size = 64
    config.epochs = 80
    config.center_crop_fraction = 1.0
    config.augment = 2
    config.scale = 0.25
    models_list = [
        dilated_32x_odd,  # parameterized(recipe_32x_odd),
        # parameterized(recipe_51x_odd),
        # parameterized(recipe_64x_odd),
        # parameterized(recipe_99x_odd),
        # parameterized(recipe_73x_odd),
    ]

    for m in models_list:
        for base_width in [16, 32]:
            hparams = {
                'base_width': base_width,
                'class_weights': 'none',
                'aug_level': config.augment,
                'ds_bg_samples': config.dataset_size,
                'scale': config.scale,
                'crop_fraction': config.center_crop_fraction,
            }
            run(m, hparams)
