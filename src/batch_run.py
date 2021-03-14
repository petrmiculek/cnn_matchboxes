import os

import tensorflow as tf

from run import run
import models

if __name__ == '__main__':
    # run(models.fcn_residual_32x_18l, augment=False)
    # run(models.fcn_residual_32x_18l, use_small_ds=True, augment=True)
    #   13-14 night
    # run(models.residual_64x, augment=True)
    # run(models.fcn_residual_32x_18l, augment=True)
    #   model-building-err
    run(models.residual_64x, use_small_ds=False, augment=False)
    # run(models.residual_32x_31l_concat, use_small_ds=True, augment=True)
