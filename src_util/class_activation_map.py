import csv
import os
from math import sqrt

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from labels import load_labels, img_dims
from image_regions import crop_out, get_boundaries

import sys

if not sys.warnoptions:
    import warnings

    # openCV QT warning - current thread is not object thread
    warnings.simplefilter("ignore")

# path = os.path.join(input_folder, '20201020_121210.jpg')

# debugging
class_names = ['background',
               'corner-bottom',
               'corner-top',
               'edge-bottom',
               'edge-side',
               'edge-top'
               'intersection-side',
               'intersection-top']


def show_pred(model, batch, class_names):
    batch = np.array(batch)
    pred_raw = model.predict(batch)
    #     predictions_raw = model.predict(tf.convert_to_tensor(imgs, dtype=tf.float32))
    preds = np.argmax(pred_raw, axis=1)

    for idx, pred in enumerate(preds):
        label = class_names[pred]
        fig = plt.imshow(batch[idx].astype("uint8"))

        fig.axes.axis("off")
        fig.axes.set_title('label: {}'.format(label))
        fig.axes.figure.show()


def single_image(model, path):
    """
    Get Heatmap showing region classifications
    (manual sliding window approach)
    did not work well

    :param model:
    :param path:
    :return:
    """
    from image_regions import crop_out, get_boundaries

    region_side = 64
    scale = 0.5

    img = cv.imread(path)
    img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # did not reverse indices
    dim = np.array((img.shape[0], img.shape[1]))

    # diagonal = np.linspace(
    #     (region_side, region_side),
    #     (dim[0] - region_side, dim[1] - region_side),
    #     100) \
    #     .astype(int)

    # cv.namedWindow(':)', cv.WINDOW_GUI_NORMAL)

    step_size = 64

    # generate center points
    # memo: radius = region_side // 2  => regions don't reach edges
    grid = np.mgrid[
           region_side:dim[0] - region_side:step_size, \
           region_side:dim[1] - region_side:step_size] \
        .reshape(2, -1).T

    canvas = np.zeros((len(class_names), *dim))  # channels (== classes), x, y
    # print(canvas.shape)
    for i, center in enumerate(grid):
        x1, y1, x2, y2 = get_boundaries(img, center)

        cutout = crop_out(img, x1, y1, x2, y2)

        cutout_batch = np.expand_dims(cutout, axis=0)

        if cutout.shape != (64, 64, 3):
            print(cutout.shape, cutout_batch.shape)
            print(center, '->', x1, y1, x2, y2)
            continue

        pred_raw = model.predict(cutout_batch)
        pred = np.argmax(pred_raw)
        try:
            canvas[
            pred,
            x1:x2,
            y1:y2] += 1
        except Exception as exc:
            print(exc)
            continue

    canvas[canvas > 255] = 255

    # activation map per class
    fig, axes = plt.subplots(ncols=1, nrows=8,
                             constrained_layout=True,
                             figsize=(16, 2)
                             )

    for i, name in enumerate(class_names):
        axes[i] = plt.subplot(1, 8, i + 1)
        axes[i] = plt.imshow(canvas[i].astype("uint8").T)
        axes[i].axes.axis("off")
        axes[i].axes.set_title(name)

    fig.show()

    return canvas
