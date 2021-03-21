import os
import sys

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# hacky, just for profiling
sys.path.extend(['/home/petrmiculek/Code/light_matches',
                 '/home/petrmiculek/Code/light_matches/src',
                 '/home/petrmiculek/Code/light_matches/src_util'])

from run import run, tf_init
import models

if __name__ == '__main__':
    tf_init()
    """ model runs """
    # run(models.fcn_residual_32x_18l, augment=False)
    # run(models.fcn_residual_32x_18l, use_small_ds=True, augment=True)

    #   13-14 overnight
    # run(models.residual_64x, augment=True)
    # run(models.fcn_residual_32x_18l, augment=True)
    #   model-building-err

    #   14-15 overnight
    # model_kwargs = {'skip_branch_conv': False}
    # run(models.residual_64x, model_kwargs, use_small_ds=False, augment=False)
    #
    # run(models.residual_64x, use_small_ds=False, augment=True)
    #
    # run(models.residual_64x_33l_concat, use_small_ds=False, augment=False)

    # run(models.residual_32x_31l_concat, use_small_ds=True, augment=True)

    #   15-16 overnight
    # run(models.dilated_32x_exp2, use_small_ds=True, augment=True)
    # run(models.dilated_32x_exp2, use_small_ds=False, augment=False)
    # run(models.dilated_32x_exp2, use_small_ds=False, augment=True)

    #   failed
    # run(models.dilated_64x_exp2, use_small_ds=False, augment=False)
    # run(models.dilated_64x_exp2, use_small_ds=False, augment=True)

    #   20

    for width in [8, 16]:
        for pooling_freq in [1, 2]:
            for pooling in ['max', 'None']:
                model_kwargs = {'width': width,
                                'pooling': pooling,
                                'pooling_freq': pooling_freq}

                run(models.dilated_64x_exp2, use_small_ds=True, augment=True, model_kwargs=model_kwargs)

    for width in [8, 16]:
        model_kwargs = {'width': width,
                        'pooling': 'max',
                        'pooling_freq': 1}

        run(models.dilated_64x_exp2, use_small_ds=False, augment=True, model_kwargs=model_kwargs)

