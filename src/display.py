# stdlib
import os

# external
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# local
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
