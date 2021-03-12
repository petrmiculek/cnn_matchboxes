
import tensorflow as tf

from run import run

if __name__ == '__main__':
    run(models.fcn_residual_1, use_small_ds=True, augment=False)
    run(models.fcn_residual_1, use_small_ds=True, augment=True)
    run(models.fcn_residual_1, use_small_ds=True, augment=True)
