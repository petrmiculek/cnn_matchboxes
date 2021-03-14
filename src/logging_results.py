import csv
import json
import os
from os.path import isfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model


def log_model_info(model, output_location=None):
    """Save model architecture plot (image) and model config (json)

    :param model:
    :param output_location:
    :return:
    """
    if len(model.layers) == 2:
        aug = model.layers[0]
        base_model = model.layers[1]
        aug.summary()

    else:
        print('log_model_info: unexpected model structure')
        base_model = model

    base_model.summary()

    plot_model(base_model, os.path.join(output_location, base_model.name + "_architecture.png"), show_shapes=True)

    try:
        with open(os.path.join(output_location, 'model_config.json'), mode='x') as json_out:
            json_out.write(model.to_json())
    except FileExistsError:
        print('Model config already exists, did not overwrite.')


def log_full_img_pred_losses(model_name, img_path, losses_sum, category_losses):
    """Log full-image prediction losses (=error) to csv

    :param model_name:
    :param img_path:
    :param losses_sum:
    :param category_losses:
    :return:
    """
    def write_or_append_to_file(path, content):
        mode = 'a' if isfile(csv_sum) else 'w'

        with open(path, mode) as csv_file:
            csv_writer = csv.writer(csv_file)  # , delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
            csv_writer.writerow(content)

    csv_sum = 'outputs/losses_sum.csv'
    csv_cat = 'outputs/losses_categories.csv'

    write_or_append_to_file(csv_sum, [model_name, img_path, losses_sum])
    write_or_append_to_file(csv_cat, [model_name, img_path, *category_losses])
