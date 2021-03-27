import csv
import os
from collections import defaultdict
import numpy as np
import pandas as pd


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


def load_labels_pandas(path, use_full_path=True, keep_bg=True):
    csv = pd.read_csv(path,
                      header=None,
                      names=['category', 'x', 'y', 'image', 'img_x', 'img_y'],
                      dtype={'category': str, 'x': np.int32, 'y': np.int32,
                             'image': str, 'img_x': np.int32, 'img_y': np.int32}
                      )
    if use_full_path:
        csv['image'] = os.path.dirname(path) + os.sep + csv['image']

    if not keep_bg:
        csv = csv[csv.category != 'background']

    return csv
