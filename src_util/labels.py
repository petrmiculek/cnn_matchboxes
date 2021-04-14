import csv
import os
from collections import defaultdict
import numpy as np
import pandas as pd

from general import lru_cache


@lru_cache(copy=True)
def load_labels_dict(path, use_full_path=True, keep_bg=True):
    """Load annotated points

    :param path: path to labels file
    :param use_full_path: file_name includes file path
    :param keep_bg:
    :return: nested dict: file_name, categories, list of (x, y)
    """

    # dict of file_names, dict of categories, list of (x, y)
    labels = defaultdict(lambda: defaultdict(list))

    with open(path) as f:  # os.curdir + os.sep + path + os.sep +
        csv_data = csv.reader(f, delimiter=',')
        for line in csv_data:
            if len(line) == 0:
                continue

            label_name, x, y, img_name, img_w, img_h = line

            if use_full_path:
                img_name = os.path.dirname(path) + os.sep + img_name

            if label_name == 'background' and not keep_bg:
                continue

            img = labels[img_name]
            img[label_name].append((int(x), int(y)))

    return labels


@lru_cache(copy=True)
def load_labels(path, use_full_path=True, keep_bg=True):
    csv = pd.read_csv(path,
                      header=None,
                      names=['category', 'x', 'y', 'image', 'img_x', 'img_y'],
                      dtype={'category': str, 'x': np.int32, 'y': np.int32,
                             'image': str, 'img_x': np.int32, 'img_y': np.int32}
                      )
    if use_full_path:
        csv['image'] = os.path.join(os.path.dirname(path), csv['image'])

    if not keep_bg:
        csv = csv[csv.category != 'background']

    return csv


def rescale_labels(labels, scale, model_crop_delta, center_crop_fraction):
    """

    Fractional crop
    Scale
    Model crop
    + culling points outside cropped areas

    :param labels: labels dataframe
    :param scale:
    :param model_crop_delta:
    :param center_crop_fraction:
    :return:
    """
    labels = labels.copy()

    img_size = labels.iloc[0][['img_x', 'img_y']].to_numpy()  # assume same-sized images

    center_crop = ((1 - center_crop_fraction) / 2 * img_size).astype(np.int)

    # cull points outside center-cropped area
    labels = labels[(center_crop[0] <= labels.x) &
                    (center_crop[1] <= labels.y)]
    labels = labels[(labels.x <= (img_size[0] - center_crop[0])) &
                    (labels.y <= (img_size[1] - center_crop[1]))]

    img_size -= center_crop

    # center-crop
    labels.x -= center_crop[0]
    labels.y -= center_crop[1]

    # initial rescale
    labels.x = (labels.x * scale).astype(np.int)
    labels.y = (labels.y * scale).astype(np.int)

    img_size = (img_size * scale).astype(np.int)

    # cull points outside model-cropped area
    model_crop = model_crop_delta // 2

    labels = labels[(model_crop <= labels.x) &
                    (model_crop <= labels.y)]
    labels = labels[(labels.x <= (img_size[0] - model_crop)) &
                    (labels.y <= (img_size[1] - model_crop))]

    # model-crop
    labels.x -= model_crop
    labels.y -= model_crop
    # although the model_crop_delta is always odd, the floor division does

    img_size -= model_crop

    print(img_size)  # prediction should be just as big

    return labels


def rescale_labels_dict(dict_labels, orig_img_size, scale, model_crop_delta, center_crop_fraction):
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
