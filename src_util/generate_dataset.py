# stdlib
import argparse
import csv
import os
import random
import sys

# external
import cv2 as cv
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# local
from src_util.labels import load_labels_dict

"""
Dataset prep:

padding x radius for background

"""

"""
Example standalone usage:
## 64x
# bg100
python src_util/generate_dataset.py -f -b -c 64 -p 100 -s 50
python src_util/generate_dataset.py -b -c 64 -r -p 100 -s 50
python src_util/generate_dataset.py -f -b -c 64 -p 100 -s 50 -v
python src_util/generate_dataset.py -b -c 64 -r -p 100 -s 50 -v

# bg500
python src_util/generate_dataset.py -f -b -c 64 -p 500 -s 50
python src_util/generate_dataset.py -b -c 64 -r -p 500 -s 50
python src_util/generate_dataset.py -f -b -c 64 -p 500 -s 50 -v
python src_util/generate_dataset.py -b -c 64 -r -p 500 -s 50 -v

## 128x
# bg100
python src_util/generate_dataset.py -f -b -c 128 -p 100 -s 50
python src_util/generate_dataset.py -b -c 128 -r -p 100 -s 50
python src_util/generate_dataset.py -f -b -c 128 -p 100 -s 50 -v
python src_util/generate_dataset.py -b -c 128 -r -p 100 -s 50 -v

# bg500
python src_util/generate_dataset.py -f -b -c 128 -p 500 -s 50
python src_util/generate_dataset.py -b -c 128 -r -p 500 -s 50
python src_util/generate_dataset.py -f -b -c 128 -p 500 -s 50 -v
python src_util/generate_dataset.py -b -c 128 -r -p 500 -s 50 -v

# bg250
python src_util/generate_dataset.py -f -b -c 64 -p 250 -s 50
python src_util/generate_dataset.py -b -c 64 -r -p 250 -s 50
python src_util/generate_dataset.py -f -b -c 64 -p 250 -s 50 -v
python src_util/generate_dataset.py -b -c 64 -r -p 250 -s 50 -v
"""

random.seed(1234)
np.random.seed(1234)


def get_boundaries(img, center_pos, radius=32):
    """

    :param img:
    :param center_pos: [x, y]
    :param radius:
    :return: top-left and bottom-right coordinates for region
    """

    # numpy image
    dim_x = img.shape[0]
    dim_y = img.shape[1]

    # value clipping
    x = max(center_pos[0], radius)
    x = min(x, dim_x - radius - 1)

    y = max(center_pos[1], radius)
    y = min(y, dim_y - radius - 1)

    return x - radius, y - radius, x + radius, y + radius


def cut_out_around_point(img, center_pos, radius=32):
    low_x, low_y, high_x, high_y = get_boundaries(img, center_pos, radius)
    try:
        cropped = img[low_x:high_x, low_y:high_y].copy()  # x,y
    except Exception as ex:
        print(ex)
        print(img is None, '{}-{}, {}-{}'.format(low_x, high_x, low_y, high_y))
        cropped = None

    return cropped


def images_to_dataset(do_foreground=True, do_background=True, val=False, region_side=64,
                      per_image_samples=100, scale_percentage=50, reduced_sampling_area=False, output_folder=None):
    """Create keypoint and/or background samples from images and annotations

    side-note: there are 1700 annotations in ~50 images => 35 keypoints per image
    """
    # fixed parameter
    min_dist_bg_from_kp = 16

    scale = scale_percentage / 100
    assert region_side % 2 == 0, "Use even cutout size"
    radius = region_side // 2

    padding = int((4032 * scale) / 5)
    # radius -> full image, padding -> center
    if reduced_sampling_area:
        cutout_padding = padding
    else:
        cutout_padding = radius

    # in/out folders
    input_folder = 'sirky' + '_val' * val

    if output_folder is None:
        run_params = '{}x_{:03d}s_{}bg'.format(region_side, scale_percentage, per_image_samples)
        output_path = 'datasets' + os.sep + run_params + '_val' * val
    else:
        output_path = output_folder

    labels_file = input_folder + os.sep + 'labels.csv'
    bg_csv_file = output_path + os.sep + 'background' + '.csv'
    merged_labels_bg_csv = output_path + os.sep + 'labels_with_bg' + '.csv'

    if not os.path.isdir(input_folder) or \
            not os.path.isfile(labels_file):
        print('could not find input folders/files')
        sys.exit(0)

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    print('saving to', output_path)

    labels = load_labels_dict(labels_file, use_full_path=False)

    for file in labels:  # dict of labelled_files
        """
        Coordinate conventions:
        CV -> y,x
        NumPy -> x,y

        img = cv
        labels = x,y = np

        everything is done in (x,y)-space up to the image-cropout itself (which is called with y,x)
        """
        img = cv.imread(input_folder + os.sep + file)  # -> y,x
        orig_size = img.shape[1], img.shape[0]  # y,x -> x,y
        img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)  # -> y,x

        if do_foreground:
            # generating regions from labels = keypoints
            for category in labels[file]:  # dict of categories
                category_folder = output_path + os.sep + category

                if not os.path.isdir(category_folder):
                    os.makedirs(category_folder, exist_ok=True)

                for label_pos in labels[file][category]:  # list of labels
                    label_pos_scaled = np.array(label_pos, dtype=np.intc) * scale  # x,y
                    label_pos_scaled = label_pos_scaled.astype(dtype=np.intc)

                    region = cut_out_around_point(img, (label_pos_scaled[1], label_pos_scaled[0]), radius)

                    # save image
                    filename_no_suffix = file.split('.')[0]
                    region_filename = '{}_({},{}).jpg'.format(filename_no_suffix, label_pos[0], label_pos[1])

                    tmp = cv.imwrite(category_folder + os.sep + region_filename, region)

        file_labels = [label
                       for cat in labels[file]
                       for label in labels[file][cat]]  # -> x,y

        file_labels_scaled = (np.array(file_labels, dtype=np.intc) * scale).astype(np.intc)

        if do_background:
            # generating regions from the background
            category = 'background'
            if not os.path.isdir(output_path + os.sep + category):
                os.makedirs(output_path + os.sep + category, exist_ok=True)

            # generate more samples than needed, some might get filtered due to proximity
            samples = int(4 * per_image_samples)

            # try normal distribution
            # scipy.stats.norm.fit(file_labels).rvs(samples).astype(np.int)

            mins = np.min(file_labels_scaled, axis=0)
            maxs = np.max(file_labels_scaled, axis=0)

            min_x = np.min([mins[0], cutout_padding])
            max_x = np.max([maxs[0], img.shape[1] - cutout_padding])

            min_y = np.min([mins[1], cutout_padding])
            max_y = np.max([maxs[1], img.shape[0] - cutout_padding])

            coords = np.vstack([
                np.random.randint(min_x, max_x, samples),
                np.random.randint(min_y, max_y, samples)
            ]).T  # <- note the .T

            coords = np.unique(coords, axis=0)

            assert min_x >= 0
            assert min_y >= 0

            # debug
            # coords = np.vstack([
            #     np.array([min_x, max_x]),
            #     np.array([min_y, max_y]),
            # ]).T.astype(np.intc)

            min_dists = cdist(coords, file_labels_scaled).min(axis=1)
            indices = np.where(min_dists > min_dist_bg_from_kp, True, False)
            coords = coords[indices]

            np.random.shuffle(coords)

            if len(coords) < per_image_samples:
                print('Too few points that are far enough({})'.format(len(coords)), file=sys.stderr)

            coords = coords[:per_image_samples]

            if np.any(coords < 0):
                print('Warning: negative coordinates in background samples')

            # save background positions to a csv file (same structure as keypoints)
            with open(bg_csv_file, 'a') as csvfile:
                out_csv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

                for pos in coords:  # how many background samples from image
                    region_background = cut_out_around_point(img, (pos[1], pos[0]), radius)
                    region_path = output_path + os.sep + category + os.sep  # per category folder

                    # since positions were generated already re-scaled, de-scale them when writing
                    x, y = (pos / scale).astype(np.intc)

                    filename_no_suffix = file.split('.')[0]
                    region_filename = '{}_({},{}).jpg'.format(filename_no_suffix, x, y)

                    cv.imwrite(region_path + region_filename, region_background)
                    out_csv.writerow(['background', x, y, file, orig_size[0], orig_size[1]])

    if do_background:
        if not os.path.isfile(bg_csv_file):
            print('no background points csv file, skipping creation of merged labels')
        else:
            # Merge keypoints + background csv for cutout positions visualization
            labels_csv = pd.read_csv(labels_file, header=None)
            bg_csv = pd.read_csv(bg_csv_file, header=None)
            merged = pd.concat([labels_csv, bg_csv], ignore_index=True)

            merged.to_csv(merged_labels_bg_csv, header=False, index=False, encoding='utf-8')
            # todo warning - manually created background labels won't be in background.csv


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

    parser.add_argument('--output_folder', '-o', type=str, default=None,
                        help="output folder")

    # python src_util/generate_dataset.py -f -b -c 64 -p 100 -s 50
    # python src_util/generate_dataset.py -b -c 64 -r -p 100 -s 50

    args = parser.parse_args()
    images_to_dataset(args.foreground, args.background, args.val, args.cutout_dize,
                      args.per_image_samples, args.scale_percentage, args.reduced_sampling_area, args.output_folder)


