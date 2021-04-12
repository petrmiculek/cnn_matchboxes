import os
import sys
from math import floor, ceil, log10

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import cdist

import run_config
from general import lru_cache
from labels import load_labels_pandas
from logging_results import log_mean_square_error_csv
from show_results import display_predictions


def get_image_as_batch(img_path, scale=1.0, center_crop_fraction=1.0):
    """Open and Process image

    :param img_path: Path for loading image
    :param scale: Rescale image
    :param center_crop_fraction: Crop out rectangle with sides equal to a given fraction of original image
    :return:
    """

    img = cv.imread(img_path)

    orig_size = img.shape[1], img.shape[0], img.shape[2]  # reversed indices

    if center_crop_fraction != 1.0:
        # crop from center
        low = (1 - center_crop_fraction) / 2
        high = (1 + center_crop_fraction) / 2
        img = img[int(img.shape[0] * low): int(img.shape[0] * high),
                  int(img.shape[1] * low): int(img.shape[1] * high),
                  :]

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                    interpolation=cv.INTER_AREA)  # reversed indices, OK
    img_batch = np.expand_dims(img, 0)
    return img_batch, orig_size


def make_prediction(base_model, img, maxes_only=False, undecided_only=False):
    """Make full-image prediction

    Not worth trying to turn this into tf.ops, since it requires further workarounds for assigning to tensors

    :param base_model:
    :param img: img as batch
    :param maxes_only:
    :param undecided_only:
    :return:
    """

    predictions_raw = base_model.predict(tf.convert_to_tensor(img, dtype=tf.uint8))
    predictions = tf.squeeze(predictions_raw).numpy()

    if maxes_only and not undecided_only:
        maxes = np.zeros(predictions.shape)
        max_indexes = np.argmax(predictions, axis=-1)
        maxes[np.arange(predictions.shape[0])[:, None],  # in every row
              np.arange(predictions.shape[1]),  # in every column
              max_indexes] = 1  # the maximum-predicted category (=channel) is set to 1

        predictions = maxes

    if undecided_only and not maxes_only:
        decisive_values = np.max(predictions, axis=2)  # per-pixel
        decisive_indices = np.where(decisive_values < 0.5, True, False)
        decisive_indices = np.stack((decisive_indices,) * 8, axis=-1)
        undecided = np.where(decisive_indices, predictions, 0)

        predictions = undecided

    return predictions


def crop_to_prediction(img, pred):
    model_crop_delta = img.shape[0] - pred.shape[0]
    low = floor(model_crop_delta / 2)
    high = ceil(model_crop_delta / 2)

    return img[low:-high, low:-high], model_crop_delta


@lru_cache
def grid_like(x, y):
    grid = np.meshgrid(np.arange(x), np.arange(y))
    grid = np.vstack([grid])
    grid = np.transpose(grid, (1, 2, 0))
    return grid


def rescale_file_labels_dict(dict_labels, orig_img_size, scale, model_crop_delta, center_crop_fraction):
    """Unused"""
    new = dict()
    for cat, labels in dict_labels.items():
        new_l = []
        for pos in labels:
            p = int(pos[0]) * scale, \
                int(pos[1]) * scale  # e.g. 3024 -> 1512

            center_crop_diff = orig_img_size[0] * scale * (1 - center_crop_fraction) // 2, \
                               orig_img_size[1] * scale * (1 - center_crop_fraction) // 2

            p = p[0] - center_crop_diff[0], \
                p[1] - center_crop_diff[1]  # e.g. 1512 - 378 -> 1134

            p = p[0] - model_crop_delta // 2, \
                p[1] - model_crop_delta // 2
            new_l.append(p)
        new[cat] = new_l

    return new


def rescale_file_labels_pandas(file_labels, orig_img_size, scale, model_crop_delta, center_crop_fraction):
    # initial rescale
    file_labels.x = (file_labels.x * scale).astype(np.int32)
    file_labels.y = (file_labels.y * scale).astype(np.int32)

    # center-crop
    file_labels.x -= int(orig_img_size[0] * scale * (1 - center_crop_fraction) // 2)
    file_labels.y -= int(orig_img_size[1] * scale * (1 - center_crop_fraction) // 2)

    # model-crop
    file_labels.x -= model_crop_delta // 2
    file_labels.y -= model_crop_delta // 2

    return file_labels


def mean_square_error(predictions, img_path, file_labels, class_names, base_model):
    """Calculate MSE metric for full-image prediction

    Metric is calculated per-pixel.
    Consequences:
        Different scale factor is accounted for -> sum / scale**2
        Different model crop rate is not accounted for:
        32x -> -31,
        128x -> -127

        Also, distance from keypoint will be lower at smaller scales.

        Center crop fraction also makes it easier.

    todo calculate MSE from prediction probability instead of argmax?

    for all prediction-pixels:
        non-background: distance to closest keypoint of given class ^ 2
                        if no GT keypoint of given category present, fixed penalty

        background: 0.0, if distance to nearest keypoint > min_dist_background
                    else fixed penalty

    """
    min_dist_background = 2.0
    min_dist_background_penalty = 1e3
    no_such_cat_penalty = 1e6

    file_labels_merged = np.array([file_labels.x, file_labels.y]).T

    # make prediction one-hot over channels
    p_argmax = np.argmax(predictions, axis=2)

    # grid for calculating distances
    grid = grid_like(predictions.shape[1], predictions.shape[0])

    category_losses = {}

    """ background """
    bg_mask = np.where(p_argmax == 0, True, False)  # 2D mask
    bg_list = grid[bg_mask]  # list of 2D coords

    per_pixel_distances = cdist(bg_list, file_labels_merged)
    per_pixel_loss = np.min(per_pixel_distances, axis=1)
    penalized = per_pixel_loss[per_pixel_loss < min_dist_background]
    category_losses['background'] = min_dist_background_penalty * len(penalized) / run_config.scale**2

    """ keypoint categories """
    for i, cat in enumerate(class_names[1:], start=1):  # skip 0 == background
        # filter predictions of given category
        cat_list = grid[p_argmax == i]
        # cat_mask = np.where(p_argmax == i, True, False)  # 2D mask

        if cat not in file_labels.category.values:  # no GT labels of this category
            # predicted non-present category
            category_losses[cat] = len(cat_list) * no_such_cat_penalty / run_config.scale**2
            continue

        # distance of each predicted point to each GT label
        cat_labels = file_labels[file_labels.category == cat]
        cat_labels = np.array([cat_labels.x, cat_labels.y]).T
        per_pixel_distances = np.square(cdist(cat_list, cat_labels))
        per_pixel_loss = np.min(per_pixel_distances, axis=1)

        category_losses[cat] = np.sum(per_pixel_loss) / run_config.scale**2

    loss_values_only = list(category_losses.values())
    loss_sum = np.sum(loss_values_only)

    # log to csv
    log_mean_square_error_csv(base_model.name, img_path, loss_sum, loss_values_only)

    return loss_sum, \
           [str(cat) + ': 1e{0:0.2g}'.format(log10(loss + 1)) for cat, loss in category_losses.items()]


def full_prediction(base_model, class_names, file_labels, img_path, output_location=None,
                    show=True, maxes_only=False, undecided_only=False):
    """Show predicted heatmaps for full image

    :param base_model:
    :param class_names:
    :param file_labels:
    :param img_path:
    :param output_location:
    :param show:
    :param maxes_only: Show only the top predicted category per point.
    :param undecided_only: Show only predictions that were
    """

    img, orig_size = get_image_as_batch(img_path, run_config.scale, run_config.center_crop_fraction)

    predictions = make_prediction(base_model, img, maxes_only, undecided_only)
    img = img[0]  # remove batch dimension

    # Model prediction is cropped, adjust image accordingly
    img, model_crop_delta = crop_to_prediction(img, predictions)

    category_titles = class_names

    losses_sum = -1
    if undecided_only:
        plots_title = 'undecided'
    elif maxes_only:
        plots_title = 'maxes'
    else:
        try:
            file_labels = rescale_file_labels_pandas(file_labels, orig_size, run_config.scale, model_crop_delta,
                                                     run_config.center_crop_fraction)

            losses_sum, category_losses = mean_square_error(predictions, img_path, file_labels, class_names,
                                                            base_model)
            category_titles = category_losses
            plots_title = str(losses_sum // 1e6) + 'M'

            show_mse_location(predictions, img, img_path, file_labels, class_names,
                              output_location=output_location, show=False)
        except Exception as ex:
            plots_title = ''
            print(ex, file=sys.stderr)

    # save_predictions_cv(predictions, img, img_path, output_location)
    display_predictions(predictions, img, img_path, category_titles, plots_title,
                        show=show, output_location=output_location, superimpose=True)

    return losses_sum


def full_prediction_all(base_model, val=True,
                        undecided_only=False, maxes_only=False,
                        output_location=None, show=True):
    labels_dir = 'sirky' + '_val' * val
    labels = load_labels_pandas(labels_dir + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)

    if output_location:
        output_location = os.path.join(output_location, 'heatmaps')
        os.makedirs(output_location, exist_ok=True)

    mse_list = []
    for file in pd.unique(labels['image']):
        file_labels = labels[labels.image == file]
        mse = full_prediction(base_model,
                              run_config.class_names,
                              file_labels,
                              img_path=labels_dir + os.sep + file,
                              output_location=output_location,
                              show=show,
                              undecided_only=undecided_only,
                              maxes_only=maxes_only)
        mse_list.append(mse)

    avg_mse = np.mean(np.array(mse_list))
    return avg_mse


def show_mse_location(predictions, img, img_path, file_labels, class_names, output_location=None, show=True):
    """Show contributions to MSE in full image predictions

    :param predictions:
    :param img:
    :param img_path:
    :param file_labels:
    :param class_names:
    :param show:
    :param output_location:
    :return:

    inspiration for grid operations (4-th answer):
    https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy

    idea: grid distance to keypoint could be cached and loaded from a pickle

    """
    if not show and output_location is None:
        return

    def dst_pt2grid(pt):
        return np.square(np.hypot(gx - pt[0], gy - pt[1]))

    gx, gy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    flat_penalty = np.zeros((img.shape[0], img.shape[1])) + 100

    cat_distances = []
    for cat in class_names:
        if cat not in file_labels.category.values:
            if cat == 'background':
                cat_distances.append(np.zeros((img.shape[0], img.shape[1])))
            else:
                cat_distances.append(flat_penalty)
            continue

        cat_labels = file_labels[file_labels.category == cat]
        cat_labels = np.array([cat_labels.x, cat_labels.y]).T

        distance_layers = list(map(dst_pt2grid, cat_labels))
        distance_layers = np.vstack([distance_layers])
        distance_layers = distance_layers / np.sqrt(np.max(distance_layers))

        min_distance = np.min(distance_layers, axis=0)  # minimum distance to any keypoint
        cat_distances.append(min_distance)

    cat_distances = np.dstack(cat_distances)
    titles = np.append(class_names, 'full_image')

    # elementwise multiply predictions with cat_distances
    tst = predictions * cat_distances

    # tst expected to be in [0 .. 1] range,
    tst /= np.max(tst)
    # but it would be barely visible like that
    tst *= 10

    if output_location:
        output_location = os.path.join(output_location, 'err_distribution')

    # show/save with matplotlib
    display_predictions(tst, img, img_path, titles, output_location=output_location, show=show)
    # save with OpenCV - unused
    # save_predictions_cv(tst, img, img_path, output_location)
