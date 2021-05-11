# stdlib
import os

# external
from math import ceil, floor, sqrt

import PIL
import numpy as np
import cv2 as cv
import tensorflow as tf
from matplotlib import pyplot as plt

# local
import config
from src_util.general import safestr


# disable profiling-related-errors
# def profile(x):
#     return x


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
def display_predictions(predictions, img, img_path, class_titles, title='',
                        show=True, output_location=None, superimpose=False):
    """Plot predictions as heatmaps superimposed on input image

    :param predictions: model predictions WxHx8
    :param img: image for which prediction was made, W'xH'x3
    :param img_path: image path
    :param class_titles: per-category titles (class names, per-category error)
    :param title: plot title
    :param show: show interactive/sciView plot
    :param output_location: save output (=where to save)
    :param superimpose: overlay prediction over image
    """

    if not show and output_location is None:
        return

    class_activations = postprocess_predictions(img, predictions, superimpose)

    subplot_titles = np.append(class_titles, 'full-image')

    # Plot heatmaps
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))  # 16, 14
    fig.suptitle('Heatmaps\n{}'.format(title))
    fig.subplots_adjust(right=0.85, left=0.05)
    for i in range(9):
        ax = axes[i // 3, i % 3]
        ax.imshow(class_activations[i])
        ax.axis('off')
        ax.set_title(subplot_titles[i])

    fig.tight_layout()
    if show:
        fig.show()

    if output_location:
        os.makedirs(output_location, exist_ok=True)
        fig_location = name_image_saving(img_path, output_location, 'prediction') + '.png'
        fig.savefig(fig_location, bbox_inches='tight')
    plt.close(fig)


def display_keypoints(keypoints_categories, img, img_path, class_titles, title='',
                        show=True, output_location=None, superimpose=False):
    """Plot predicted keypoints on input image

    :param keypoints_categories: model predictions WxHx8
    :param img: image for which prediction was made, W'xH'x3
    :param img_path: image path
    :param class_titles: per-category titles (class names, per-category error)
    :param title: plot title
    :param show: show interactive/sciView plot
    :param output_location: save output (=where to save)
    :param superimpose: unused, only for making args consistent with display_predictions
    """

    if not show and output_location is None:
        return

    keypoints, categories = keypoints_categories

    subplot_titles = np.append(class_titles, 'full-image')

    # Plot heatmaps
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))  # 16, 14
    fig.suptitle('Keypoints\n{}'.format(title))
    fig.subplots_adjust(right=0.85, left=0.05)
    for i in range(9):
        ax = axes[i // 3, i % 3]
        ax.imshow(img)
        kp = keypoints[categories == i]
        if kp.size > 0:
            ax.scatter(kp[:, 0], kp[:, 1], marker='.', color='orange')
        ax.axis('off')
        ax.set_title(subplot_titles[i])

    fig.tight_layout()
    if show:
        fig.show()

    if output_location:
        os.makedirs(output_location, exist_ok=True)
        fig_location = name_image_saving(img_path, output_location, 'keypoints') + '.png'
        fig.savefig(fig_location, bbox_inches='tight')
    plt.close(fig)


def name_image_saving(img_path, output_location, prefix='heatmap'):
    # sane location + filename, NO SUFFIX
    img_path_no_suffix = safestr(img_path[0:img_path.rfind('.')])
    fig_location = os.path.join(output_location, f'{prefix}_{img_path_no_suffix}')
    return fig_location


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


def plot_mse_history():
    import pandas as pd
    import os
    import matplotlib.pyplot as plt

    history = pd.read_csv(os.path.join('outputs', 'losses_sum.csv'))


def plot_class_weights(class_names, y_data, y_label, title):
    import seaborn as sns
    import matplotlib.pyplot as plt

    palette = sns.color_palette('Blues_d', len(class_names))  # [::-1]

    if np.all(y_data != y_data[0]):
        ranks = np.argsort(y_data).argsort()
    else:
        ranks = np.ones_like(y_data, dtype=np.int) + len(y_data) // 2

    palette = list(np.array(palette)[ranks])

    with sns.color_palette(palette, len(class_names)):
        sns.set_context('talk')
        figure = plt.figure(figsize=(10, 8))
        sns.set_style('darkgrid')
        gridspec = figure.add_gridspec(1, 1)
        ax = figure.add_subplot(gridspec[0, 0])
        sns.barplot(x=class_names, y=y_data, palette=palette)
        ax.set_yscale('log')
        ax.set_ylabel(y_label)
        ax.set_xlabel('class')
        ax.set_ylim(1 / 100, 100)

        figure.suptitle(title)

    for item in ax.get_xticklabels():
        item.set_rotation(90)

    figure.tight_layout()
    from general import safestr
    title = safestr(title)
    figure.savefig(f'dataset_{title}.pdf', bbox_inches='tight')
    figure.show()


def generate_class_weights_plots(class_names, class_counts, class_weights):
    from datasets import weights_inverse_freq, weights_effective_number
    # uniform weights
    plot_class_weights(class_names, np.ones(len(class_names)), 'weight', 'No weighting')

    # inverse frequency weights
    inv = np.array(list(weights_inverse_freq(class_counts).values()))
    plot_class_weights(class_names, inv, 'weight', 'Inverse Frequency')

    # effective number of samples
    for f in [1, 10, 100, 1000, 10000, 100000]:
        b = 1 - 1/f

        eff = np.array(list(weights_effective_number(class_counts, b).values()))
        plot_class_weights(class_names, eff, 'weight', f'Effective Number of Samples[Beta=(N-1)/N]')


def show_layer_activations(model, data_augmentation, ds, show=True, output_location=None):
    """Predict single cutout and show network's layer activations

    Adapted from:
    https://towardsdatascience.com/feature-visualization-on-convolutional-neural-networks-keras-5561a116d1af

    :param output_location:
    :param show:
    :param data_augmentation:
    :param model:
    :param ds:
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

    print('GT   =', config.class_names[int(labels[idx].numpy())])
    all_layer_activations = model_all_outputs(batch_img0, training=False)
    pred = all_layer_activations[-1].numpy()
    predicted_category = config.class_names[np.argmax(pred)]
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

        scale = 1.0 / size
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


def show_augmentation(data_augmentation, imgs):
    """Show grid of augmentation results

    :param data_augmentation:
    :param dataset:
    :return:
    """

    """Convert dataset"""
    # todo use predict_all_tf, with added training param
    # imgs = [img
    #         for batch in list(dataset)
    #         for img in batch[0]]
    #
    # imgs = np.vstack([imgs])  # -> 4D [img_count x width x height x channels]

    """Make predictions"""
    preds = []
    for img in imgs:
        img = img[None, ...]
        pred = data_augmentation(tf.convert_to_tensor(img, dtype=tf.uint8), training=True)
        preds.append(pred[0])

    preds = np.stack(preds, axis=0)

    low = 1 * imgs.shape[2] // 4
    high = 3 * imgs.shape[2] // 4

    # Center-crop regions (64x to 32x)
    imgs = imgs[:, low:high, low:high, :]

    # alternatively, resize to 1/2 inside the loop below, once merged (B, W, H, C) -> (B*W, H, C)
    # imgs_col = cv.resize(imgs_col, None, fx=0.5, fy=0.5)

    """Show predictions as a grid"""
    # rows = floor(sqrt(len(imgs)))  # Note: does not use all images
    rows = 8
    cols = []

    for i in range(len(imgs) // rows):
        # make a column [W, N * H]
        imgs_col = np.hstack(imgs[rows * i:(i + 1) * rows])
        pred_col = np.hstack(preds[rows * i:(i + 1) * rows])
        col = np.vstack((imgs_col, pred_col))
        cols.append(col)

    grid_img = np.vstack(cols)

    pi = PIL.Image.fromarray(grid_img.astype('uint8'))
    pi.show()
