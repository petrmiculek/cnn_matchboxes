import cv2 as cv
import numpy as np
import csv
from collections import defaultdict
import pathlib
import os
from labels import load_labels, img_dims
import random
from math import sqrt

# global parameters
region_side = 32
scale = 0.25
input_folder = 'sirky'
output_folder = 'image_regions_{}_{:03d}'.format(region_side, int(scale * 100))
labels_file = 'labels.csv'

radius = region_side // 2
random.seed(1234)


# def random_color():
#     return randint(0, 255), randint(0, 255), randint(0, 255)


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


def crop_out_from_center(img, center_pos):
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
    for file in labels:  # dict of labelled_files

        img = cv.imread(input_folder + os.sep + file)
        img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # reversed indices, OK
        file_labels = []

        mean_label_per_category_count = np.mean([len([labels[file][category] for category in labels[file]])])

        # generating regions from labels = keypoints
        for category in labels[file]:  # dict of categories
            if not os.path.isdir(output_folder + os.sep + category):
                os.makedirs(output_folder + os.sep + category, exist_ok=True)

            for label_pos in labels[file][category]:  # list of labels
                label_pos_scaled = int(int(label_pos[1]) * scale), int(int(label_pos[0]) * scale)
                # ^ inner int() does parsing, not rounding

                file_labels.append(label_pos_scaled)

                region = crop_out_from_center(img, label_pos_scaled)

                # save image
                region_path = output_folder + os.sep + category + os.sep  # per category folder
                region_filename = file.split('.')[0] + '_(' + str(label_pos[0]) + ',' + str(label_pos[1]) + ').jpg'

                tmp = cv.imwrite(region_path + region_filename, region)

        # generating regions from the background
        category = 'background'
        if not os.path.isdir(output_folder + os.sep + category):
            os.makedirs(output_folder + os.sep + category, exist_ok=True)

        repeated = 0
        for i in np.arange(np.ceil(mean_label_per_category_count)):
            pos = (0, 0)
            repeat = True
            while repeat:
                repeat = False
                # generate position (region-center)
                pos = int(random.randint(radius, img.shape[0] - radius)), \
                      int(random.randint(radius, img.shape[1] - radius))
                # todo removed scaling here

                # test that position is not too close to an existing label
                for label_pos in file_labels:
                    # file_labels positions are already properly scaled
                    if euclid_dist(pos, label_pos) < region_side:
                        repeated += 1
                        repeat = True

            region_background = crop_out_from_center(img, pos)
            region_path = output_folder + os.sep + category + os.sep  # per category folder
            region_filename = file.split('.')[0] + '_(' + str(pos[0]) + ',' + str(pos[1]) + ').jpg'

            tmp = cv.imwrite(region_path + region_filename, region_background)

        # how many generated positions were wrong (= too close to keypoints)
        # print(repeated)
