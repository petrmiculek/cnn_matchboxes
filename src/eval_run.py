# stdlib
import os

# external
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# local
import run_config
import model_ops
import models
from datasets import get_dataset
from eval_images import full_prediction_all
from eval_samples import evaluate_model, show_layer_activations


if __name__ == '__main__':
    run_config.train = False
    run_config.dataset_size = 1000
    run_config.center_crop_fraction = 0.5

    # depends on model
    # train dim decided by model
    run_config.scale = 0.5
    run_config.dataset_dim = 64

    # should not matter
    run_config.augment = True
    run_config.use_weights = False

    run_config.output_location = None
    run_config.show = True

    # load_model_name = 'dilated_64x_exp2_2021-03-26-04-57-57_full'  # /data/datasets/64x_050s_500bg
    # load_model_name = 'dilated_64x_exp2_2021-03-27-06-36-50_full'  # /data/datasets/128x_050s_500bg
    # load_model_name = 'dilated_32x_exp22021-03-28-16-38-11_full'  # /data/datasets/64x_050s_500bg
    # load_model_name = 'dilated_64x_exp2_2021-03-29-15-58-47_full'  # /data/datasets/128x_050s_1000bg
    load_model_name = 'dilated_64x_exp2_2021-03-29-15-58-47_full'  # /data/datasets/128x_050s_1000bg
    run_config.model_name = load_model_name + '_reloaded'
    for dim in [32, 64, 128]:
        if '{}x'.format(dim) in load_model_name:
            run_config.train_dim = dim
            run_config.dataset_dim = 2 * dim
            break

    base_model, model, aug_model = model_ops.load_model(load_model_name, load_weights=True)
    # callbacks = model_ops.get_callbacks()
    run_config.epochs_trained = 123  #

    dataset_dir = f'/data/datasets/{run_config.dataset_dim}x_{int(100 * run_config.scale):03d}s_{run_config.dataset_size}bg'

    val_ds, _, _ = get_dataset(dataset_dir + '_val')
    train_ds, run_config.class_names, _ = get_dataset(dataset_dir, use_weights=run_config.use_weights)

    if False:
        avg_mse_val = full_prediction_all(base_model, val=True, output_location=run_config.output_location,
                                          show=run_config.show)

        avg_mse_train = full_prediction_all(base_model, val=False, output_location=run_config.output_location,
                                            show=run_config.show)

        val_accu = evaluate_model(model, val_ds, val=True, output_location=run_config.output_location,
                                      show=run_config.show, misclassified=False)

        evaluate_model(model, train_ds, val=False, output_location=run_config.output_location, show=run_config.show)

        val_metrics = model.evaluate(val_ds, verbose=0)
        train_metrics = model.evaluate(train_ds, verbose=0)

        show_layer_activations(base_model, aug_model, val_ds, show=False,
                                       output_location=run_config.output_location)
