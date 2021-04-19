# stdlib
import os

# external
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local
import config
import model_ops
import models
from datasets import get_dataset
from eval_images import full_prediction_all
from eval_samples import evaluate_model, show_layer_activations


if __name__ == '__main__':
    config.train = False
    config.dataset_size = 1000
    config.center_crop_fraction = 0.5

    # depends on model
    # train dim decided by model
    config.scale = 0.5
    config.dataset_dim = 64

    # should not matter
    config.augment = True
    config.use_weights = False

    config.output_location = None
    config.show = True

    # load_model_name = 'dilated_64x_exp2_2021-03-26-04-57-57_full'  # /data/datasets/64x_050s_500bg
    # load_model_name = 'dilated_64x_exp2_2021-03-27-06-36-50_full'  # /data/datasets/128x_050s_500bg
    # load_model_name = 'dilated_32x_exp22021-03-28-16-38-11_full'  # /data/datasets/64x_050s_500bg
    # load_model_name = 'dilated_64x_exp2_2021-03-29-15-58-47_full'  # /data/datasets/128x_050s_1000bg
    load_model_name = 'dilated_64x_exp2_2021-03-29-15-58-47_full'  # /data/datasets/128x_050s_1000bg
    config.model_name = load_model_name + '_reloaded'
    for dim in [32, 64, 128]:
        if '{}x'.format(dim) in load_model_name:
            config.train_dim = dim
            config.dataset_dim = 2 * dim
            break

    base_model, model, aug_model = model_ops.load_model(load_model_name, load_weights=True)
    # callbacks = model_ops.get_callbacks()
    config.epochs_trained = 123  #

    dataset_dir = f'/data/datasets/{config.dataset_dim}x_{int(100 * config.scale):03d}s_{config.dataset_size}bg'

    val_ds, _, _ = get_dataset(dataset_dir + '_val')
    train_ds, config.class_names, _ = get_dataset(dataset_dir, use_weights=config.use_weights)

    if False:
        mse_pix_val, mse_val, count_mae_val = \
            full_prediction_all(base_model, val=True, output_location=config.output_location, show=config.show)

        mse_pix_train, mse_train, count_mae = \
            full_prediction_all(base_model, val=False, output_location=config.output_location, show=config.show)

        val_accu = evaluate_model(model, val_ds, val=True, output_location=config.output_location,
                                      show=config.show, misclassified=False)

        evaluate_model(model, train_ds, val=False, output_location=config.output_location, show=config.show)

        val_metrics = model.evaluate(val_ds, verbose=0)
        train_metrics = model.evaluate(train_ds, verbose=0)

        show_layer_activations(base_model, aug_model, val_ds, show=False,
                                       output_location=config.output_location)
