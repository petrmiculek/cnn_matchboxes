import cv2 as cv
import numpy as np
import csv
from collections import defaultdict
import pathlib
import os
from labels import load_labels, img_dims
from random import randint
from math import sqrt

region_side = 64


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)


def draw_cross(img, center_pos, line_length=20, color=(255, 0, 0), width=4):
    global scale
    width = max(int(width * scale), 1)
    line_length = int(line_length * scale)
    x, y = center_pos[0], center_pos[1]

    cv.line(img, (x - line_length, y - line_length), (x + line_length, y + line_length), color, width)
    cv.line(img, (x + line_length, y - line_length), (x - line_length, y + line_length), color, width)


def crop_out(img, center_pos, side=None):
    if side is None:
        side = region_side

    radius = side // 2
    x, y = center_pos[0], center_pos[1]

    # numpy image [y, x]
    dim_x = img.shape[1]
    dim_y = img.shape[0]

    # value clipping
    if center_pos[0] - side < 0:
        x = radius

    if center_pos[1] - side < 0:
        y = radius

    if center_pos[0] + side >= dim_x:
        x = dim_x - radius - 1

    if center_pos[1] + side >= dim_y:
        y = dim_y - radius - 1

    # crop
    try:
        cropped = img[y - radius: y + radius, x - radius: x + radius].copy()
    except IndexError as ex:
        print(ex)
        cropped = None

    return cropped


# def random_offset(center_pos, maxoffset=10):
#     return center_pos[0] + randint(-maxoffset, maxoffset), center_pos[1] + randint(-maxoffset, maxoffset)

def euclid_dist(pos1, pos2):
    return sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


if __name__ == '__main__':
    scale = 1

    input_folder = 'sirky'
    output_folder = "image_regions"
    labels_file = 'labels.csv'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    labels = load_labels(input_folder + os.sep + labels_file, use_full_path=False)
    for file in labels:  # dict of labelled_files

        img = cv.imread(input_folder + os.sep + file)
        img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # reversed indices, OK
        file_labels = []
        # generating regions from labels = keypoints
        for category in labels[file]:  # dict of categories
            if not os.path.isdir(output_folder + os.sep + category):
                os.makedirs(output_folder + os.sep + category, exist_ok=True)

            for label_pos in labels[file][category]:  # list of labels
                label_pos_scaled = int(int(label_pos[0]) * scale), int(int(label_pos[1]) * scale)
                file_labels.append(label_pos_scaled)
                # generate region proposals
                region = crop_out(img, label_pos_scaled)

                # save image
                region_path = output_folder + os.sep + category + os.sep  # per category folder
                region_filename = file.split('.')[0] + '_(' + str(label_pos[0]) + ',' + str(label_pos[1]) + ').jpg'

                tmp = cv.imwrite(region_path + region_filename, region)
    # generating regions from the background

    # category = 'background'
    # if not os.path.isdir(output_folder + os.sep + category):
    #     os.makedirs(output_folder + os.sep + category, exist_ok=True)
    #
    # for i in range(500):
    #     pos = (0, 0)
    #     repeat = True
    #     while repeat:
    #         repeat = False
    #         # generate position
    #         pos = \
    #             int(randint(region_side // 2, img.shape[1] - region_side // 2) * scale), \
    #             int(randint(region_side // 2, img.shape[0] - region_side // 2) * scale)
    #
    #         # test that position is not too close to an existing label
    #         for label_pos in file_labels:
    #             if euclid_dist(pos, label_pos) < (region_side // 2):
    #                 repeat = True
    #
    #     region_background = crop_out(img, pos)
    #     region_path = output_folder + os.sep + category + os.sep  # per category folder
    #     region_filename = file.split('.')[0] + '_(' + str(pos[0]) + ',' + str(pos[1]) + ').jpg'
    #
    #     tmp = cv.imwrite(region_path + region_filename, region_background)
    #     print('.', end='')
