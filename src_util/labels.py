import csv
import os
from collections import defaultdict

img_dims = dict()


def load_labels(path, use_full_path=True):
    """


    :param path: path to labels file
    :param use_full_path: file_name includes file path
    :return: nested dict: file_name, categories, list of (x, y)
    """

    # dict of file_names, dict of categories, list of (x, y)
    labels = defaultdict(lambda: defaultdict(list))

    global img_dims

    with open(path) as f:  # os.curdir + os.sep + path + os.sep +
        csv_data = csv.reader(f, delimiter=',')
        for line in csv_data:
            if len(line) == 0:
                continue

            label_name, x, y, img_name, img_w, img_h = line

            if use_full_path:
                img_name = path + os.sep + img_name

            img = labels[img_name]
            img[label_name].append((x, y))

            img_dims[img_name] = (img_w, img_h)  # overwrites old value, risky

    return labels
