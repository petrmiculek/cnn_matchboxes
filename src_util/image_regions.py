import argparse
import csv
import os
import pathlib
import random
import sys
from math import sqrt
from collections import defaultdict

import cv2 as cv
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist

from labels import load_labels

"""
Dataset prep:

padding x radius for background

todo cdist instead of euclid_dist
"""

"""
Example usage:
## 64x
# bg100
python src_util/image_regions.py -f -b -c 64 -p 100 -s 50
python src_util/image_regions.py -b -c 64 -r -p 100 -s 50
python src_util/image_regions.py -f -b -c 64 -p 100 -s 50 -v
python src_util/image_regions.py -b -c 64 -r -p 100 -s 50 -v

# bg250
python src_util/image_regions.py -f -b -c 64 -p 250 -s 50
python src_util/image_regions.py -b -c 64 -r -p 250 -s 50
python src_util/image_regions.py -f -b -c 64 -p 250 -s 50 -v
python src_util/image_regions.py -b -c 64 -r -p 250 -s 50 -v

# bg500
python src_util/image_regions.py -f -b -c 64 -p 500 -s 50
python src_util/image_regions.py -b -c 64 -r -p 500 -s 50
python src_util/image_regions.py -f -b -c 64 -p 500 -s 50 -v
python src_util/image_regions.py -b -c 64 -r -p 500 -s 50 -v

## 128x
# bg100
python src_util/image_regions.py -f -b -c 128 -p 100 -s 50
python src_util/image_regions.py -b -c 128 -r -p 100 -s 50
python src_util/image_regions.py -f -b -c 128 -p 100 -s 50 -v
python src_util/image_regions.py -b -c 128 -r -p 100 -s 50 -v
"""

random.seed(1234)
np.random.seed(1234)


def crop_out(img, x1, y1, x2, y2):
    try:
        cropped = img[x1:x2, y1:y2].copy()
    except Exception as ex:
        print(ex)
        print(img is None, x1, y1, x2, y2)
        cropped = None

    return cropped


def get_boundaries(img, center_pos, radius=32):
    """

    :param img: [y, x]
    :param center_pos: [x, y]
    :param radius:
    :return: top-left and bottom-right coordinates for region
    """

    # numpy image [y, x]
    dim_x = img.shape[1]
    dim_y = img.shape[0]

    # value clipping
    x = max(center_pos[0], radius)
    x = min(x, dim_x - radius - 1)

    y = max(center_pos[1], radius)
    y = min(y, dim_y - radius - 1)

    return x - radius, y - radius, x + radius, y + radius


def cut_out_around_point(img, center_pos, radius=32):
    x1, x2, y1, y2 = get_boundaries(img, center_pos, radius)
    return crop_out(img, x1, x2, y1, y2)


# not useful for training
# def random_offset(center_pos, maxoffset=10):
#     return center_pos[0] + randint(-maxoffset, maxoffset), center_pos[1] + randint(-maxoffset, maxoffset)


def euclid_dist(pos1, pos2):
    return sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--foreground', '-f', action='store_true',
                        help="create foreground=keypoint samples")

    parser.add_argument('--background', '-b', action='store_true',
                        help="create background samples")

    parser.add_argument('--val', '-v', action='store_true',
                        help="use validation data (instead of training)")

    parser.add_argument('--cutout_size', '-c', type=int, default=64,
                        help="cutout=sample size")

    parser.add_argument('--per_image_samples', '-p', type=int, default=100,
                        help="background sample count")

    parser.add_argument('--scale_percentage', '-s', type=int, default=50,
                        help="scaling percentage")

    parser.add_argument('--reduced_sampling_area', '-r', action='store_true',
                        help="generate background samples closer to the image center")

    # python src_util/image_regions.py -f -b -c 64 -p 100 -s 50
    # python src_util/image_regions.py -b -c 64 -r -p 100 -s 50

    args = parser.parse_args()

    scale = args.scale_percentage / 100
    region_side = args.cutout_size
    radius = args.cutout_size // 2

    padding = (4032 * scale) // 3
    # radius -> full image, padding -> center
    if args.reduced_sampling_area:
        cutout_padding = padding
    else:
        cutout_padding = radius

    # in/out folders
    input_folder = 'sirky' + '_val' * args.val
    run_params = f'{args.cutout_size}x_{args.scale_percentage:03d}s_{args.per_image_samples}bg'
    output_path = 'datasets' + os.sep + run_params + '_val' * args.val

    labels_file = input_folder + os.sep + 'labels.csv'
    bg_csv_file = output_path + os.sep + 'background' + '.csv'
    merged_labels_bg_csv = output_path + os.sep + 'labels_with_bg' + '.csv'

    if \
            not os.path.isdir(input_folder) or \
                    not os.path.isfile(labels_file):
        print('could not find input folders/files')
        sys.exit(0)

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    labels = load_labels(labels_file, use_full_path=False)
    # mean_label_per_category_count = np.mean([len(labels[file]) for file in labels])
    # there are 1700 annotations in ~50 images => 35 keypoints per image
    # can do 100 background points per image

    for file in labels:  # dict of labelled_files

        img = cv.imread(input_folder + os.sep + file)
        orig_size = img.shape[1], img.shape[0]
        img = cv.resize(img,
                        (int(img.shape[1] * scale),
                         int(img.shape[0] * scale)),
                        interpolation=cv.INTER_AREA)  # reversed indices, OK

        if args.foreground:
            # generating regions from labels = keypoints
            for category in labels[file]:  # dict of categories
                category_folder = output_path + os.sep + category

                if not os.path.isdir(category_folder):
                    os.makedirs(category_folder, exist_ok=True)

                for label_pos in labels[file][category]:  # list of labels
                    label_pos_scaled = int(int(label_pos[1]) * scale), int(int(label_pos[0]) * scale)
                    # ^ inner int() does parsing, not rounding

                    region = cut_out_around_point(img, label_pos_scaled, radius)

                    # save image
                    region_filename = file.split('.')[0] + '_(' + str(label_pos[0]) + ',' + str(label_pos[1]) + ').jpg'

                    tmp = cv.imwrite(category_folder + os.sep + region_filename, region)

        file_labels = [label
                       for cat in labels[file]
                       for label in labels[file][cat]]

        file_labels_scaled = (np.array(file_labels, dtype=np.intc) * scale).astype(np.intc)

        if args.background:
            # generating regions from the background
            category = 'background'
            if not os.path.isdir(output_path + os.sep + category):
                os.makedirs(output_path + os.sep + category, exist_ok=True)

            samples = 2 * args.per_image_samples  # generate more than needed, some might get filtered

            # try normal distribution?
            # scipy.stats.norm.fit(file_labels).rvs(samples).astype(np.intc)

            if args.reduced_sampling_area:
                mean_offset = (file_labels_scaled.mean(axis=0) - np.array(orig_size) / 2).astype(np.intc)
            else:
                mean_offset = [0, 0]

            coords = np.vstack([
                np.random.randint(cutout_padding, img.shape[0] - cutout_padding, samples) + mean_offset[0],
                np.random.randint(cutout_padding, img.shape[1] - cutout_padding, samples) + mean_offset[1]
            ]).T  # <- note the .T

            min_dists = cdist(coords, file_labels_scaled).min(axis=1)
            indices = np.where(min_dists > region_side // 4, True, False)
            coords = coords[indices]

            if len(coords) < args.per_image_samples:
                print('Too few points that are far enough({})'.format(len(coords)), file=sys.stderr)

            coords = coords[:args.per_image_samples]

            print('\t', mean_offset)
            print(coords)
            # continue

            # save background positions to a csv file (same structure as keypoints)
            with open(bg_csv_file, 'a') as csvfile:
                out_csv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

                for pos in coords:  # how many background samples from image
                    region_background = cut_out_around_point(img, pos, radius)
                    region_path = output_path + os.sep + category + os.sep  # per category folder

                    # since positions were generated already re-scaled, de-scale them when writing
                    x, y = str(int(pos[1] / scale)), str(int(pos[0] / scale))

                    region_filename = file.split('.')[0] + '_(' + x + ',' + y + ').jpg'

                    tmp = cv.imwrite(region_path + region_filename, region_background)
                    out_csv.writerow(['background', x, y, file, orig_size[0], orig_size[1]])

    if args.background:
        if not os.path.isfile(bg_csv_file):
            print('no background points csv file, skipping creation of merged labels')
        else:
            # Merge keypoints + background csv for cutout positions visualization
            labels_csv = pd.read_csv(labels_file, header=None)
            bg_csv = pd.read_csv(bg_csv_file, header=None)
            merged = pd.concat([labels_csv, bg_csv], ignore_index=True)

            merged.to_csv(merged_labels_bg_csv, header=False, index=False, encoding='utf-8')
            # todo warning - manually created background labels won't be in background.csv
