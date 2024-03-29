# stdlib
import csv
import os
import sys
from collections import defaultdict

# external
import numpy as np
import pandas as pd

# local
from util import lru_cache


@lru_cache(copy=True)
def load_labels(path, use_full_path=True, keep_bg=True):
    """Load image annotations from a csv file

    :param path: path to labels file
    :param use_full_path: file_name includes file path
    :param keep_bg: keep background samples
    :return: Pandas DataFrame
    """
    labels = pd.read_csv(path,
                         header=None,
                         names=['category', 'x', 'y', 'image', 'img_x', 'img_y'],
                         dtype={'category': str, 'x': np.int32, 'y': np.int32,
                                'image': str, 'img_x': np.int32, 'img_y': np.int32}
                         )
    if use_full_path:
        labels['image'] = os.path.dirname(path) + os.sep + labels['image']

    if not keep_bg:
        labels = labels[labels.category != 'background']

    return labels


def resize_labels(labels, scale, model_crop_delta, center_crop_fraction):
    """Transform labels' coordinates (as a Pandas DataFrame) according to run-configuration

    Fractional crop
    Scale
    Model crop
    + culling points outside cropped areas

    :param labels: labels DataFrame
    :param scale:
    :param model_crop_delta:
    :param center_crop_fraction:
    :return: DataFrame with labels' positions transformed
    """
    labels = labels.copy()

    img_size = labels[['img_x', 'img_y']].to_numpy()
    assert np.array_equal(np.min(img_size, axis=0), np.max(img_size, axis=0)), \
        "Image sizes not equal"

    img_size = img_size[0]

    center_crop = ((1 - center_crop_fraction) / 2 * img_size).astype(np.int)

    # cull points outside center-cropped area
    labels = labels[(center_crop[0] <= labels.x) &
                    (center_crop[1] <= labels.y)]
    labels = labels[(labels.x <= (img_size[0] - center_crop[0])) &
                    (labels.y <= (img_size[1] - center_crop[1]))]

    img_size -= 2 * center_crop

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

    img_size -= model_crop_delta

    # print(img_size)  # this must match with prediction size -- checked, it does

    return labels


@lru_cache(copy=True)
def load_labels_dict(path, use_full_path=True, keep_bg=True):
    """Load image annotations from a csv file

    unused

    :param path: path to labels file
    :param use_full_path: file_name includes file path
    :param keep_bg: keep background samples
    :return: nested dict: file_name -> categories, category -> list of (x, y)
    """

    # dict of file_names, dict of categories, list of (x, y)
    labels = defaultdict(lambda: defaultdict(list))

    with open(path) as f:
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


def resize_labels_dict(dict_labels, orig_img_size, scale, model_crop_delta, center_crop_fraction):
    """Transform labels' coordinates (as a dictionary) according to run-configuration

    unused

    :param dict_labels: labels dictionary
    :param orig_img_size:
    :param scale:
    :param model_crop_delta:
    :param center_crop_fraction:
    :return:
    """
    new = dict()
    for cat, labels in dict_labels.items():
        new_l = []
        for pos in labels:
            p = int(pos[0]) * scale, \
                int(pos[1]) * scale

            center_crop_diff = orig_img_size[0] * scale * (1 - center_crop_fraction) // 2, \
                               orig_img_size[1] * scale * (1 - center_crop_fraction) // 2

            p = p[0] - center_crop_diff[0], \
                p[1] - center_crop_diff[1]

            p = p[0] - model_crop_delta // 2, \
                p[1] - model_crop_delta // 2
            new_l.append(p)
        new[cat] = new_l

    return new


@lru_cache
def get_gt_counts_all(counts_path=None):
    if counts_path is None:
        counts_path = os.path.join('sirky', 'count.txt')

    if not os.path.isfile(counts_path):
        raise OSError(f'GT Counts file "{counts_path}" does not exist')

    counts_gt = pd.read_csv(counts_path,
                            header=None,
                            names=['image', 'cnt'],
                            dtype={'image': str, 'cnt': np.int32})
    return counts_gt


def get_gt_count(img_path):
    counts_gt = get_gt_counts_all()

    try:
        file = img_path[img_path.rfind(os.sep) + 1:]  # does nothing when '/' is not found
        crate_count_gt = np.array(counts_gt[counts_gt.image == file].cnt)[0]
    except Exception as exc:
        print('Reading GT count failed:\n' + str(exc), file=sys.stderr)
        crate_count_gt = -1

    return crate_count_gt
