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
def display_predictions(predictions, img, img_path, class_titles, title='', show=True,
                        output_location=None, superimpose=False):
    # Plot predictions as heatmaps superimposed on input image
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
        fig_location = name_image_saving(img_path, output_location) + '.png'
        fig.savefig(fig_location, bbox_inches='tight')
    plt.close(fig)


def name_image_saving(img_path, output_location):
    # sane location + filename, NO SUFFIX
    img_path_no_suffix = safestr(img_path[0:img_path.rfind('.')])
    fig_location = os.path.join(output_location, 'heatmap_{}'.format(img_path_no_suffix))
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
