import os

import tensorflow as tf

from run import run
import models

if __name__ == '__main__':
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
    run(models.dilated_32x_exp2, use_small_ds=True, augment=True)
    run(models.dilated_32x_exp2, use_small_ds=False, augment=False)
    run(models.dilated_32x_exp2, use_small_ds=False, augment=True)

    run(models.dilated_64x_exp2, use_small_ds=False, augment=False)
    run(models.dilated_64x_exp2, use_small_ds=False, augment=True)
