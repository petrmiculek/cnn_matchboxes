import cv2 as cv
import numpy as np
import csv
from collections import defaultdict
import pathlib
import os
import random
from math import sqrt
import matplotlib.pyplot as plt
from labels import load_labels, img_dims
from image_regions import crop_out

import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

region_side = 64
scale = 0.5
input_folder = 'sirky'

path = os.path.join(input_folder, '20201020_121210.jpg')


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


def single_image(path):
    global scale
    global region_side
    img = cv.imread(path)
    img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # reversed indices, OK
    dim = np.array((img.shape[1], img.shape[0]))

    # diagonal = np.linspace(
    #     (region_side, region_side),
    #     (dim[0] - region_side, dim[1] - region_side),
    #     100) \
    #     .astype(int)

    # cv.namedWindow(':)', cv.WINDOW_GUI_NORMAL)

    grid = np.mgrid[
           region_side:dim[0] - region_side:10, \
           region_side:dim[1] - region_side:10] \
        .reshape(2, -1).T

    batch = []
    for i, pos in enumerate(grid):
        cutout = crop_out(img, pos)
        batch.append(cutout)

        if i == 31:
            break

        if i % 32 == 31:
            show_pred(model, batch, class_names)
            batch = []

        # cv.imshow(':)', cutout)
        # k = cv.waitKey(0)

    # cv.destroyAllWindows()

    # todo
    # save predictions in a grid -> activation map
