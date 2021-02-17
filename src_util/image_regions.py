import cv2 as cv
import numpy as np
import csv
from collections import defaultdict
import pathlib
import os
from labels import load_labels, img_dims
import random
from math import sqrt

import pandas as pd

# global parameters
region_side = 32
scale = 0.50
input_folder = 'sirky'  # _validation
labels_file = 'labels.csv'
output_folder = 'image_regions_{}_{:03d}'.format(region_side, int(scale * 100))  # _val
bg_csv_file = 'background'  # _val
merged_labels_bg_csv = 'labels_with_bg'  # _val

# swapping of generating keypoint/background cutouts
do_foreground = True
do_background = True

val = False
if val:
    input_folder += '_validation'
    output_folder += '_val'
    bg_csv_file += '_val'
    merged_labels_bg_csv += '_val'

radius = region_side // 2
random.seed(1234)


# different from show_images_labelled -- investigate
def draw_cross(img, center_pos, line_length=20, color=(255, 0, 0), width=4):
    global scale
    width = max(int(width * scale), 1)
    line_length = int(line_length * scale)
    x, y = center_pos[0], center_pos[1]

    cv.line(img, (x - line_length, y - line_length), (x + line_length, y + line_length), color, width)
    cv.line(img, (x + line_length, y - line_length), (x - line_length, y + line_length), color, width)


def crop_out(img, x1, y1, x2, y2):
    try:
        cropped = img[x1:x2, y1:y2].copy()
    except IndexError as ex:
        print(ex)
        cropped = None

    return cropped


def get_boundaries(img, center_pos):
    """

    :param center_pos: [x, y]
    :param img: [y, x]
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


def cut_out_around_point(img, center_pos):
    x1, x2, y1, y2 = get_boundaries(img, center_pos)
    return crop_out(img, x1, x2, y1, y2)


# not useful for training
# def random_offset(center_pos, maxoffset=10):
#     return center_pos[0] + randint(-maxoffset, maxoffset), center_pos[1] + randint(-maxoffset, maxoffset)


def euclid_dist(pos1, pos2):
    return sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


if __name__ == '__main__':

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    labels = load_labels(input_folder + os.sep + labels_file, use_full_path=False)
    # mean_label_per_category_count = np.mean([len(labels[file]) for file in labels])
    # there are 1700 annotations in ~50 images => 35 keypoints per image
    # can do 100 background points per image

    for file in labels:  # dict of labelled_files

        img = cv.imread(input_folder + os.sep + file)
        orig_size = img.shape[1], img.shape[0]
        img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # reversed indices, OK
        file_labels = []

        if do_foreground:
            # generating regions from labels = keypoints
            for category in labels[file]:  # dict of categories
                if not os.path.isdir(output_folder + os.sep + category):
                    os.makedirs(output_folder + os.sep + category, exist_ok=True)

                for label_pos in labels[file][category]:  # list of labels
                    label_pos_scaled = int(int(label_pos[1]) * scale), int(int(label_pos[0]) * scale)
                    # ^ inner int() does parsing, not rounding

                    file_labels.append(label_pos_scaled)

                    region = cut_out_around_point(img, label_pos_scaled)

                    # save image
                    region_path = output_folder + os.sep + category + os.sep  # per category folder
                    region_filename = file.split('.')[0] + '_(' + str(label_pos[0]) + ',' + str(label_pos[1]) + ').jpg'

                    tmp = cv.imwrite(region_path + region_filename, region)

        if do_background:
            # generating regions from the background
            category = 'background'
            if not os.path.isdir(output_folder + os.sep + category):
                os.makedirs(output_folder + os.sep + category, exist_ok=True)

            padding = img.shape[0] // 3
            cutout_padding = padding  # radius -> full image, padding -> center

            repeated = 0
            # save background positions to a csv file (same structure as keypoints)
            with open(bg_csv_file + '.csv', 'a') as csvfile:
                out_csv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

                for i in range(100):  # how many background samples from image
                    pos = (0, 0)
                    repeat = True
                    while repeat:
                        repeat = False
                        # generate position (region-center)
                        pos = int(random.randint(cutout_padding - 500, img.shape[0] - cutout_padding)), \
                              int(random.randint(cutout_padding, img.shape[1] - cutout_padding))

                        # test that position is not too close to an existing label
                        for label_pos in file_labels:
                            # both file_labels positions and pos are already properly scaled
                            if euclid_dist(pos, label_pos) < 5:  # or use: region_side // 4
                                repeated += 1
                                repeat = True

                    region_background = cut_out_around_point(img, pos)
                    region_path = output_folder + os.sep + category + os.sep  # per category folder

                    # since positions were generated already re-scaled, de-scale them when writing
                    x, y = str(int(pos[1] // scale)), str(int(pos[0] // scale))

                    region_filename = file.split('.')[0] + '_(' + x + ',' + y + ').jpg'

                    tmp = cv.imwrite(region_path + region_filename, region_background)
                    out_csv.writerow(['background', x, y, file, orig_size[0], orig_size[1]])

        # how many generated positions were wrong (= too close to keypoints)
        # print(repeated)

    if do_background:
        # Merge keypoints + background csv for cutout positions visualization
        labels_csv = pd.read_csv(input_folder + os.sep + labels_file, header=None)
        bg_csv = pd.read_csv(bg_csv_file + '.csv', header=None)
        merged = pd.concat([labels_csv, bg_csv], ignore_index=True)

        merged.to_csv(merged_labels_bg_csv + '.csv', header=False, index=False, encoding='utf-8')

# todo warning - manually created background labels won't be in background.csv
