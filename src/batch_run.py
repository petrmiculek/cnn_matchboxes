import os
import sys

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# hacky, just for profiling
sys.path.extend(['/home/petrmiculek/Code/light_matches',
                 '/home/petrmiculek/Code/light_matches/src',
                 '/home/petrmiculek/Code/light_matches/src_util'])

from run import run, tf_init
from models import *
from models_old import *
import run_config

# noinspection DuplicatedCode
if __name__ == '__main__':
    tf_init()
    """ model runs """
    # run(fcn_residual_32x_18l, augment=False)
    # run(fcn_residual_32x_18l, use_small_ds=True, augment=True)

    #   13-14 overnight
    # run(residual_64x, augment=True)
    # run(fcn_residual_32x_18l, augment=True)
    #   model-building-err

    #   14-15 overnight
    # model_kwargs = {'skip_branch_conv': False}
    # run(residual_64x, model_kwargs, use_small_ds=False, augment=False)
    #
    # run(residual_64x, use_small_ds=False, augment=True)
    #
    # run(residual_64x_33l_concat, use_small_ds=False, augment=False)

    # run(residual_32x_31l_concat, use_small_ds=True, augment=True)

    #   15-16 overnight
    # run(dilated_32x_exp2, use_small_ds=True, augment=True)
    # run(dilated_32x_exp2, use_small_ds=False, augment=False)
    # run(dilated_32x_exp2, use_small_ds=False, augment=True)

    #   failed
    # run(dilated_64x_exp2, use_small_ds=False, augment=False)
    # run(dilated_64x_exp2, use_small_ds=False, augment=True)

    #   20
    """
    for width in [8, 16]:
        for pooling_freq in [1, 2]:
            for pooling in ['max', 'None']:
                model_kwargs = {'width': width,
                                'pooling': pooling,
                                'pooling_freq': pooling_freq}

                run(dilated_64x_exp2, use_small_ds=True, augment=True, model_kwargs=model_kwargs)

    for width in [8, 16]:
        model_kwargs = {'width': width,
                        'pooling': 'max',
                        'pooling_freq': 1}

        run(dilated_64x_exp2, use_small_ds=False, augment=True, model_kwargs=model_kwargs)
    """
    # run(dilated_64x_exp2, use_small_ds=True, augment=False, ds_dim=64)
    # print('returned')
    # sys.exit(0)
    """
    # 25-26
    for dim in [64, 128]:
        for aug in [True]:
            for use_weights in [False]:
                model_kwargs = {'pool': 'max', 'use_weights': use_weights}
                run(dilated_64x_exp2, model_kwargs=model_kwargs, use_small_ds=False, augment=aug, ds_dim=dim)
    """

    # 27
    """
    for dim in [32]:
        for aug in [False, True]:
            for use_weights in [False]:
                for pool_after_first_conv in [False, True]:
                    for base_width in [16]:  # 32, 64
                        model_kwargs = {'pool': 'max',
                                        'use_weights': use_weights,
                                        'pool_after_first_conv': pool_after_first_conv,
                                        'base_width': base_width,
                                        }
                                        }
                        run(dilated_32x_exp2, model_kwargs=model_kwargs, use_small_ds=True, augment=aug, ds_dim=dim)

    """
    """
    # 28, 29
    # aug = True
    # base_width = 16
    # dim = 64
    # for tail_downscale in [1, 2, 4]:
    #     model_kwargs = {'pool': 'max',
    #                     'use_weights': False,
    #                     'pool_after_first_conv': False,
    #                     'base_width': base_width,
    #                     'tail_downscale': tail_downscale,
    #                     }
    #     run(dilated_32x_exp2, model_kwargs=model_kwargs, dataset_size=1000, augment=aug, ds_dim=dim)
    """

    """
    # aug = True
    # for dim in [64, 128]:
    #     for base_width in [8, 16, 32]:
    #         model_kwargs = {'pool': 'max',
    #                         'use_weights': False,
    #                         'pool_after_first_conv': True,
    #                         'base_width': base_width,
    #                         }
    #         run(dilated_64x_exp2, model_kwargs=model_kwargs, dataset_size=1000, augment=aug, ds_dim=dim)
    #
    """
    """
    
    aug = True
    base_width = 32
    dim = 64
    for tail_downscale in [1, 2, 4]:
        model_kwargs = {'pool': 'max',
                        'use_weights': False,
                        'pool_after_first_conv': False,
                        'base_width': base_width,
                        'tail_downscale': tail_downscale,
                        }
        # run(dilated_32x_odd, model_kwargs=model_kwargs, dataset_size=1000, augment=aug, ds_dim=dim)
    """

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
    hp_base_width = hp.HParam('base_width', hp.Discrete([16, 32]))

    # non-model params
    hp_ds_bg_samples = hp.HParam('ds_bg_samples', hp.Discrete([200, 700, 1000]))
    hp_augmentation = hp.HParam('augmentation', hp.Discrete([False, True]))
    hp_scale = hp.HParam('scale', hp.Discrete([0.25, 0.5]))
    hp_crop_fraction = hp.HParam('crop_fraction', hp.Discrete([0.5, 1.0]))
    # METRIC_MSE = 'mse'

    with tf.summary.create_file_writer('logs').as_default():
        hp.hparams_config(
            hparams=[
                hp_base_width,
                hp_ds_bg_samples,
                hp_augmentation,
                hp_scale,
                hp_crop_fraction,
            ],
            metrics=[hp.Metric('mse'), hp.Metric('mse_val'), hp.Metric('pr_value_val')],
        )

    for m in [parameterized(recipe_32x_odd),
              parameterized(recipe_32x_flat5),
              parameterized(recipe_32x_exp2),
              parameterized(recipe_32x_d1to5),
              ]:
        for width in hp_base_width.domain.values:
            hparams = {
                'base_width': width,

                'augmentation': run_config.augment,
                'ds_bg_samples': run_config.dataset_size,
                'scale': run_config.scale,
                'crop_fraction': run_config.center_crop_fraction,
                # 'tail_downscale': 2  # try if logging to tb fails
            }
            run(m, hparams)
