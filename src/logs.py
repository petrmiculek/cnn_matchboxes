# stdlib
import csv
import os

# external
# -


def log_model_info(model, output_location, saved_models_location):
    """Save model architecture plot (image) and model config (json)

    :param saved_models_location:
    :param model:
    :param output_location:
    :return:
    """
    from tensorflow.keras.utils import plot_model

    line_length = 100

    if len(model.layers) == 2:
        # augmentation model + base model
        aug = model.layers[0]
        base_model = model.layers[1]
        aug.summary(line_length=line_length)

    else:
        print('W: log_model_info: unexpected model structure')
        base_model = model

    base_model.summary(line_length=line_length)

    plot_model(base_model, os.path.join(output_location, base_model.name + "_architecture.png"), show_shapes=True)

    try:
        with open(os.path.join(output_location, 'model_config.json'), mode='x') as json_out:
            json_out.write(model.to_json())
        with open(os.path.join(saved_models_location, f'{model.name}.json'), mode='x') as json_out:
            json_out.write(model.to_json())
    except FileExistsError:
        print('I: Model config already exists, did not overwrite.')


def log_mean_square_error_csv(model_name, img_path, error_sum, category_losses):
    """Log MSE from full-image prediction to csv

    :param model_name:
    :param img_path:
    :param error_sum:
    :param category_losses:
    :return:
    """
    def write_or_append_to_file(path, content):
        mode = 'a' if os.path.isfile(csv_sum) else 'w'

        with open(path, mode) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(content)

    csv_sum = 'outputs/losses_sum.csv'
    csv_cat = 'outputs/losses_categories.csv'

    write_or_append_to_file(csv_sum, [model_name, img_path, error_sum])
    write_or_append_to_file(csv_cat, [model_name, img_path, *category_losses])
