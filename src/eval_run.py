# stdlib
import os

# external
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local
import config
import model_build
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

    # should not matter
    config.augment = True
    config.use_weights = False

    config.output_location = 'best_outputs'
    config.show = True

    # load_model_name = 'dilated_64x_exp2_2021-03-26-04-57-57_full'  # /data/datasets/64x_050s_500bg
    # load_model_name = 'dilated_64x_exp2_2021-03-27-06-36-50_full'  # /data/datasets/128x_050s_500bg
    # load_model_name = 'dilated_32x_exp22021-03-28-16-38-11_full'  # /data/datasets/64x_050s_500bg
    # load_model_name = 'dilated_64x_exp2_2021-03-29-15-58-47_full'  # /data/datasets/128x_050s_1000bg
    # load_model_name = '32x_d3-5-7-1-1_2021-04-19-14-25-48_full'  # /data/datasets/64x_025s_1000bg
    load_model_name = '64x_d1-3-5-7-9-11-1-1_2021-04-23-14-10-33_full'  # /data/datasets/128x_050s_1000bg
    load_model_name = '99x_d1-3-5-7-9-11-13-1_2021-04-29-11-43-25_full'  # /data/datasets/128x_050s_1000bg
    config.model_name = load_model_name + '_reloaded'

    try:
        dim = int(load_model_name.split('x_')[0])
    except IndexError:
        print('', file=sys.stderr)
        dim = 64

    config.train_dim = dim
    config.dataset_dim = 2 * dim

    # overwriting for 99x model todo
    config.train_dim = 99
    config.dataset_dim = 128

    base_model, model, aug_model = model_build.load_model(load_model_name, load_weights=True)
    # callbacks = model_ops.get_callbacks()
    config.epochs_trained = 123  #

    dataset_dir = '/data/datasets/{}x_{:03d}s_{}bg'\
        .format(config.dataset_dim, int(100 * config.scale), config.dataset_size)

    val_ds, _, _ = get_dataset(dataset_dir + '_val')
    train_ds, config.class_names, _ = get_dataset(dataset_dir)

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


    if False:
        # comparing predictions at multiple scales
        output_location = 'scale_comparison'
        os.makedirs(output_location, exist_ok=True)
        config.show = False
        show = False
        for s in np.arange(start=0.25, stop=1.26, step=0.125):
            print(s)

            config.center_crop_fraction = 0.2
            config.scale = s

            images_dir = 'sirky' + '_val' * val
            file = os.listdir(images_dir)[-1]

            for ccf in np.arange(0.10, 1.05, 0.05):
                config.center_crop_fraction = ccf
                print(ccf)

                labels = load_labels(images_dir + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)
                labels = resize_labels(labels, config.scale, config.train_dim - 1, config.center_crop_fraction)

                file_labels = labels[labels.image == file]
                pix_mse, point_mse, count_mae = \
                    full_prediction(base_model, file_labels, img_path=images_dir + os.sep + file,
                                    output_location=output_location, show=show)
