import functools
import itertools
import operator
import os
import sys
from math import ceil, floor, log10, sqrt
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import cv2 as cv
import PIL
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as conf_mat
from scipy.spatial.distance import cdist

from src_util.general import safestr, timing
from logging_results import log_mean_square_error_csv
from src_util.labels import load_labels_dict, load_labels_pandas


# disable profiling-related-errors
# def profile(x):
#     return x


# @pro file

def confusion_matrix(model_name, class_names, epochs_trained, labels,
                     predictions, output_location=None, show=True,
                     val=False, normalize=True):
    """Create and show/save confusion matrix"""
    kwargs = {}
    if normalize:
        # {'true', 'pred', 'all'}, default = None
        normalize = 'true'
        kwargs['min'] = 0.0
        kwargs['max'] = 0.0
        kwargs['fmt'] = '0.2f'
    else:
        normalize = None
        kwargs['fmt'] = 'd'

    cm = conf_mat(list(labels), list(predictions), normalize=normalize)

    fig_cm = sns.heatmap(
        cm,
        annot=True,
        xticklabels=class_names,
        yticklabels=class_names,
        # fmt='0.2f',
        # vmin=0.0,
        # vmax=1.0
        **kwargs
    )
    fig_cm.set_title('Confusion Matrix\n{} {} [e{}]'.format(model_name, 'val' if val else 'train', epochs_trained))
    fig_cm.set_xlabel("Predicted")
    fig_cm.set_ylabel("True")
    fig_cm.axis("on")
    fig_cm.figure.tight_layout(pad=0.5)

    if show:
        fig_cm.figure.show()

    if output_location:
        fig_cm.figure.savefig(os.path.join(output_location, 'confusion_matrix' + '_val' * val + '.png'),
                              bbox_inches='tight')

    plt.close(fig_cm.figure)


# @pro file

def misclassified_regions(imgs, labels, class_names, predictions,
                          false_pred, output_location=None, show=True):
    """Show misclassified regions

    # todo alternative version with grid-like output
    """

    if len(false_pred) > 500:
        print('too many misclassified regions, refusing to generate')
        return

    for i, idx in enumerate(false_pred):
        label_true = class_names[labels[idx]]
        label_predicted = class_names[predictions[idx]]

        fig = plt.imshow(imgs[idx].numpy().astype("uint8"))

        # True label x Predicted label
        fig.axes.set_title('T:{} x F:{}'.format(label_true, label_predicted))
        fig.axes.axis("off")

        if output_location:
            # noinspection PyUnboundLocalVariable
            fig_location = os.path.join(output_location, '{}_{}_x_{}'.format(i, label_true, label_predicted))
            fig.axes.figure.savefig(fig_location, bbox_inches='tight')

        if show:
            fig.axes.figure.show()

        plt.close(fig.axes.figure)


# @pro file
# noinspection PyUnreachableCode
def visualize_results(model, dataset, class_names, epochs_trained,
                      output_location=None, show=False, misclassified=False, val=False):
    """Show misclassified regions and confusion matrix

    Get predictions for whole dataset and manually evaluate the model predictions.
    Use evaluated results to save misclassified regions and to show a confusion matrix.

    :param val:
    :param model:
    :param dataset:
    :param class_names:
    :param epochs_trained:
    :param output_location:
    :param show:
    :param misclassified:
    """

    """Dataset processing"""
    false_predictions, imgs, labels, predictions_argmaxes, predictions_juice = predict_all_tf(model, dataset)

    """Misclassified regions"""
    if misclassified:
        if output_location:
            misclassified_dir = os.path.join(output_location, 'missclassified_regions')
            os.makedirs(misclassified_dir, exist_ok=True)
        else:
            misclassified_dir = None

        misclassified_regions(imgs, labels, class_names, predictions_argmaxes,
                              false_predictions, misclassified_dir, show=False)

    """Confusion matrix"""
    confusion_matrix(model.name, class_names, epochs_trained, labels,
                     predictions_argmaxes, output_location=output_location, show=show, val=val, normalize=False)

    """More metrics"""
    maxes = np.max(predictions_juice, axis=1)
    confident = len(maxes[maxes > 0.9])
    undecided = len(maxes[maxes <= 0.125])

    accuracy = 100.0 * (1 - len(false_predictions) / len(predictions_argmaxes))
    if val:
        print('validation:')

    print('Accuracy: {0:0.3g}%'.format(accuracy))
    print('Prediction types:\n',
          '\tConfident: {}\n'.format(confident),
          '\tUndecided: {}'.format(undecided))

    return accuracy

    """
    for i in undecided_idx:
        fig = plt.imshow(np.squeeze(imgs[i]).astype("uint8"))
        fig.axes.figure.show()
    """
    """
    # print all predictions
    for i, img in enumerate(predictions_juice.numpy()):
        print('{}: '.format(i), end='')
        print(["{0:0.2f}".format(k) for k in img])
    """


def predict_all_tf(base_model, dataset):
    imgs = []
    labels = []
    for batch in list(dataset):
        imgs.append(batch[0])
        labels.append(batch[1])
    imgs = tf.concat(imgs, axis=0)  # -> 4D [img_count x width x height x channels]
    labels = tf.concat(labels, axis=0)  # -> 1D [img_count]

    ds_reconstructed = tf.data.Dataset.from_tensor_slices(imgs).batch(64).prefetch(buffer_size=2)

    predictions = base_model.predict(ds_reconstructed)
    predictions_juice = tf.squeeze(predictions)
    predictions_maxes = tf.argmax(predictions_juice, axis=1)
    false_predictions = tf.squeeze(tf.where(labels != predictions_maxes))

    return false_predictions, imgs, labels, predictions_maxes, predictions_juice


def predict_all_numpy(base_model, dataset):
    # superseded by predict_all_tf
    imgs = []
    labels = []
    for batch in list(dataset):
        imgs.append(batch[0].numpy())
        labels.append(batch[1].numpy())

    imgs = np.vstack(imgs)  # -> 4D [img_count x width x height x channels]
    labels = np.hstack(labels)  # -> 1D [img_count]
    """Making predictions"""
    imgs_tensor = tf.convert_to_tensor(imgs, dtype=tf.uint8)
    ds_reconstructed = tf.data.Dataset.from_tensor_slices(imgs_tensor).batch(32).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    predictions_raw = timing(base_model.predict)(ds_reconstructed)
    predictions_juice = tf.squeeze(predictions_raw)
    predictions = tf.argmax(predictions_juice, axis=1)
    false_predictions = np.where(labels != predictions)[0]  # reduce dimensions of a nested array

    return false_predictions, imgs, labels, predictions, predictions_juice


def heatmaps_all(base_model, class_names, val=True,
                 undecided_only=False, maxes_only=False,
                 output_location=None, show=True, epochs_trained=0):
    labels_dir = 'sirky' + '_val' * val
    # labels = load_labels_dict(labels_dir + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)
    labels = load_labels_pandas(labels_dir + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)

    if output_location:
        output_location = os.path.join(output_location, 'heatmaps')
        os.makedirs(output_location, exist_ok=True)

    errs = []
    for file in pd.unique(labels['image']):
        file_labels = labels[labels.image == file]
        err = predict_full_image(base_model, class_names,
                                 file_labels,
                                 img_path=labels_dir + os.sep + file,
                                 output_location=output_location,
                                 show=show,
                                 undecided_only=undecided_only,
                                 maxes_only=maxes_only)
        errs.append(err)

    avg_mean_absolute_error = np.mean(np.array(errs))
    tf.summary.scalar('avg_mae' + '_val' * val, avg_mean_absolute_error, step=epochs_trained)


def get_image_as_batch(img_path, scale=1.0, center_crop_fraction=1.0):
    """Open and Process image

    :param center_crop_fraction:
    :param img_path:
    :param scale:
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
        decisive_values = np.max(predictions, axis=2)
        decisive_indices = np.where(decisive_values < 0.5, True, False)
        decisive_indices = np.stack((decisive_indices,) * 8, axis=-1)
        undecided = np.where(decisive_indices, predictions, 0)

        predictions = undecided

    return predictions


def full_img_mse_error(predictions, img_path, file_labels, class_names, base_model):
    """Calculate MSE metric for full-image prediction

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

    grid = np.meshgrid(np.arange(predictions.shape[1]), np.arange(predictions.shape[0]))
    grid = np.vstack([grid])
    grid = np.transpose(grid, (1, 2, 0))

    category_losses = {}

    """ background """
    bg_mask = np.where(p_argmax == 0, True, False)  # 2D mask
    bg_list = grid[bg_mask]  # list of 2D coords
    uniq = np.unique(bg_list, axis=0)
    if len(uniq) != len(bg_list):
        print(f'uniq {len(uniq)} / {len(bg_list)}')

    per_pixel_distances = cdist(bg_list, file_labels_merged)
    per_pixel_loss = np.min(per_pixel_distances, axis=1)
    penalized = per_pixel_loss[per_pixel_loss < min_dist_background]
    category_losses['background'] = min_dist_background_penalty * len(penalized)

    """ keypoint categories """
    for i, cat in enumerate(class_names[1:], start=1):  # skip 0 == background
        # filter predictions of given category
        cat_list = grid[p_argmax == i]
        # cat_mask = np.where(p_argmax == i, True, False)  # 2D mask

        if cat not in file_labels.category.values:  # no GT labels of this category
            # predicted non-present category
            category_losses[cat] = len(cat_list) * no_such_cat_penalty
            continue

        # distance of each predicted point to each GT label
        cat_labels = file_labels[file_labels.category == cat]
        cat_labels = np.array([cat_labels.x, cat_labels.y]).T
        per_pixel_distances = np.square(cdist(cat_list, cat_labels))
        per_pixel_loss = np.min(per_pixel_distances, axis=1)

        category_losses[cat] = np.sum(per_pixel_loss)

    loss_values_only = list(category_losses.values())
    loss_sum = np.sum(loss_values_only)

    # log to csv
    log_mean_square_error_csv(base_model.name, img_path, loss_sum, loss_values_only)

    return loss_sum, \
           [str(cat) + ': 1e{0:0.2g}'.format(log10(loss + 1)) for cat, loss in category_losses.items()]


def crop_to_prediction(img, pred):
    model_crop_delta = img.shape[0] - pred.shape[0]
    low = floor(model_crop_delta / 2)
    high = ceil(model_crop_delta / 2)

    return img[low:-high, low:-high], model_crop_delta


def rescale_file_labels(dict_labels, orig_img_size, scale, model_crop_delta, center_crop_fraction):
    # todo test function: show annotations on a scaled-down image

    new = dict()
    for cat, labels in dict_labels.items():
        new_l = []
        for pos in labels:
            # p = np.array(pos) * scale  # numpy way
            # p = p - center_crop_diff - model_crop_delta // 2

            p = int(pos[0]) * scale, \
                int(pos[1]) * scale  # 3024 -> 1512

            center_crop_diff = orig_img_size[0] * scale * (1 - center_crop_fraction) // 2, \
                               orig_img_size[1] * scale * (1 - center_crop_fraction) // 2

            p = p[0] - center_crop_diff[0], \
                p[1] - center_crop_diff[1]  # 1512 - 378 -> 1134

            p = p[0] - model_crop_delta // 2, \
                p[1] - model_crop_delta // 2  #
            new_l.append(p)
        new[cat] = new_l

    return new


def rescale_file_labels_pandas(file_labels, orig_img_size, scale, model_crop_delta, center_crop_fraction):
    # initial rescale
    file_labels.x = (file_labels.x * scale).astype('int32')
    file_labels.y = (file_labels.y * scale).astype('int32')

    # center-crop
    file_labels.x -= int(orig_img_size[0] * scale * (1 - center_crop_fraction) // 2)
    file_labels.y -= int(orig_img_size[1] * scale * (1 - center_crop_fraction) // 2)

    # model-crop
    file_labels.x -= model_crop_delta // 2
    file_labels.y -= model_crop_delta // 2

    return file_labels


# @pro file
def predict_full_image(base_model, class_names, file_labels, img_path, output_location=None,
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

    scale = 0.5
    center_crop_fraction = 0.5

    img, orig_size = get_image_as_batch(img_path, scale, center_crop_fraction)

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
            file_labels = rescale_file_labels_pandas(file_labels, orig_size, scale, model_crop_delta, center_crop_fraction)

            losses_sum, category_losses = full_img_mse_error(predictions, img_path, file_labels, class_names,
                                                             base_model)
            category_titles = category_losses
            plots_title = str(losses_sum // 1e6) + 'M'

            show_mse_location(predictions, img, img_path, file_labels, class_names,
                              output_location=output_location, show=show)
        except Exception as e:
            plots_title = ''
            print(e, file=sys.stderr)

    # save_predictions_cv(predictions, img, img_path, output_location)
    display_predictions(predictions, img, img_path, category_titles, plots_title,
                        show=show, output_location=output_location, superimpose=True)

    return losses_sum


# @pro file

def display_predictions(predictions, img, img_path, class_names, title='', show=True,
                        output_location=None, superimpose=False):
    # Plot predictions as heatmaps superimposed on input image

    class_activations = postprocess_predictions(img, predictions, superimpose)

    if output_location:
        os.makedirs(output_location, exist_ok=True)
        fig_location = name_image_saving(img_path, output_location) + '.png'

    subplot_titles = np.append(class_names, 'full-image')

    # Plot heatmaps
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))  # 16, 14
    fig.suptitle('Heatmaps\n{}'.format(title))
    fig.subplots_adjust(right=0.85, left=0.05)
    for i in range(9):
        ax = axes[i // 3, i % 3].imshow(class_activations[i])
        axes[i // 3, i % 3].axis('off')
        axes[i // 3, i % 3].set_title(subplot_titles[i])

    fig.tight_layout()
    if show:
        fig.show()

    if output_location:
        fig.savefig(fig_location, bbox_inches='tight')
    plt.close(fig)


def name_image_saving(img_path, output_location):
    # sane location + filename, NO SUFFIX
    img_path_no_suffix = safestr(img_path[0:img_path.rfind('.')])
    fig_location = os.path.join(output_location, 'heatmap_{}'.format(img_path_no_suffix))
    return fig_location


# @pro file

def postprocess_predictions(img, predictions, superimpose=False):
    predictions = np.uint8(255 * predictions)
    # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    class_activations = []
    heatmap_alpha = 0.7
    for i in range(8):
        pred = cv.cvtColor(predictions[:, :, i], cv.COLOR_GRAY2BGR)
        pred = cv.applyColorMap(pred, cv.COLORMAP_VIRIDIS)
        pred = cv.cvtColor(pred, cv.COLOR_BGR2RGB)
        if superimpose:
            pred = cv.addWeighted(pred, heatmap_alpha, img, 1 - heatmap_alpha, gamma=0)
        class_activations.append(pred)
    class_activations.append(img)
    return class_activations


# @pro file

def gallery(array, ncols=3):
    """
    https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    :param array:
    :param ncols:
    :return:
    """

    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # -> [H * rows, W * cols, C]
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


# @pro file

def save_predictions_cv(predictions, img, img_path, output_location):
    class_activations = postprocess_predictions(img, predictions, superimpose=False)

    # img = cv.resize(img, (predictions.shape[0], predictions.shape[1]))
    class_activations = np.stack(np.vstack([class_activations]), axis=0)

    class_activations = gallery(class_activations)

    fig_location = name_image_saving(img_path, output_location) + '.bmp'

    cv.imwrite(fig_location, class_activations)


def tile(arr, nrows, ncols):
    """
    unused

    https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy

    Args:
        arr: HWC format array
        nrows: number of tiled rows
        ncols: number of tiled columns
    """
    h, w, c = arr.shape
    out_height = nrows * h
    out_width = ncols * w
    chw = np.moveaxis(arr, (0, 1, 2), (1, 2, 0))

    if c < nrows * ncols:
        chw = chw.reshape(-1).copy()
        chw.resize(nrows * ncols * h * w)

    return (chw
            .reshape((nrows, ncols, h, w))
            .swapaxes(1, 2)
            .reshape(out_height, out_width))


# @pro file

def show_layer_activations(model, data_augmentation, ds, class_names, show=True, output_location=None):
    """Predict single cutout and show network's layer activations

    Adapted from:
    https://towardsdatascience.com/feature-visualization-on-convolutional-neural-networks-keras-5561a116d1af

    :param output_location:
    :param show:
    :param data_augmentation:
    :param model:
    :param ds:
    :param class_names:
    :return:
    """

    if output_location:
        output_location = os.path.join(output_location, 'layer_activations')
        os.makedirs(output_location, exist_ok=True)

    # choose non-background sample
    batch, labels = next(iter(ds))  # first batch
    idx = np.argmax(labels)

    batch_img0 = tf.convert_to_tensor(batch[idx: idx + 1])  # first image (made to a 1-element batch)

    layers = [layer.output for layer in model.layers]
    model_all_outputs = tf.keras.Model(inputs=model.input, outputs=layers)

    # no augmentation, only crop
    batch_img0 = data_augmentation(batch_img0, training=False)

    print('GT   =', class_names[int(labels[idx].numpy())])
    all_layer_activations = model_all_outputs(batch_img0, training=False)
    pred = all_layer_activations[-1].numpy()
    predicted_category = class_names[np.argmax(pred)]
    print('pred =', pred.shape, predicted_category)

    # I could test for region not being background here
    # ... or just read in images of known categories

    """ Show input image """
    fig_input, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(batch_img0[0].numpy().astype('uint8'), cmap='viridis', vmin=0, vmax=255)
    if output_location:
        plt.savefig(os.path.join(output_location, '00_input' + predicted_category))

    if show:
        plt.show()

    plt.close(fig_input.figure)

    layer_names = []
    for i, layer in enumerate(model.layers):
        layer_names.append(str(i) + '_' + layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, all_layer_activations):

        # debugging image scaling - how many values end up out of range
        out_of_range_count = 0
        total_count = 0

        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).

        n_cols = ceil(n_features / images_per_row)  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(min(images_per_row, n_features)):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]

                channel_image *= 64
                channel_image += 128

                # how often do we clip
                min_, max_ = np.min(channel_image), np.max(channel_image)
                if min_ < 0.0 or max_ > 255.0:
                    out_of_range_count += 1
                total_count += 1

                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        fig = plt.figure(figsize=(scale * display_grid.shape[1],
                                  scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis', vmin=0, vmax=255)

        if output_location:
            plt.savefig(os.path.join(output_location, layer_name))

        if show:
            plt.show()

        plt.close(fig.figure)

        # print(f'{layer_name}: {out_of_range_count=} / {total_count}')


# @pro file

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


# @pro file

def show_augmentation(data_augmentation, dataset):
    """Show grid of augmentation results

    :param data_augmentation:
    :param dataset:
    :return:
    """

    """Convert dataset"""
    imgs = [img
            for batch in list(dataset)
            for img in batch[0]]

    imgs = np.vstack([imgs])  # -> 4D [img_count x width x height x channels]

    """Make predictions"""
    pred = data_augmentation(tf.convert_to_tensor(imgs, dtype=tf.uint8), training=True)

    low = 1 * imgs.shape[2] // 4
    high = 3 * imgs.shape[2] // 4

    # Center-crop regions (64x to 32x)
    imgs = imgs[:, low:high, low:high, :]

    # alternatively, resize to 1/2 inside the loop below, once merged (B, W, H, C) -> (B*W, H, C)
    # imgs_col = cv.resize(imgs_col, None, fx=0.5, fy=0.5)

    """Show predictions as a grid"""
    rows = floor(sqrt(len(imgs)))  # Note: does not use all images
    cols = []

    for i in range(len(imgs) // rows):
        # make a column [W, N * H]
        imgs_col = np.vstack(imgs[rows * i:(i + 1) * rows])
        pred_col = np.vstack(pred[rows * i:(i + 1) * rows])
        col = np.hstack((imgs_col, pred_col))
        cols.append(col)

    grid_img = np.hstack(cols)

    pi = PIL.Image.fromarray(grid_img.astype('uint8'))
    pi.show()
