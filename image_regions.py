import cv2 as cv
import numpy as np
import csv
from collections import defaultdict
import pathlib
import os
from labels import load_labels, img_dims
from random import randint


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)


def draw_cross(img, center_pos, line_length=20, color=(255, 0, 0), width=4):
    global scale
    width = max(int(width * scale), 1)
    line_length = int(line_length * scale)
    x, y = center_pos[0], center_pos[1]

    cv.line(img, (x - line_length, y - line_length), (x + line_length, y + line_length), color, width)
    cv.line(img, (x + line_length, y - line_length), (x - line_length, y + line_length), color, width)


def crop_out(img, center_pos, side=64):
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


def random_offset(center_pos, maxoffset=10):
    return center_pos[0] + randint(-maxoffset, maxoffset), center_pos[1] + randint(-maxoffset, maxoffset)


if __name__ == '__main__':
    scale = 1

    input_folder = 'sirky'
    output_folder = "image_regions"
    labels_file = 'labels.csv'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    labels = load_labels(input_folder + os.sep + labels_file, use_full_path=False)

    # cv.namedWindow(':-)', cv.WINDOW_GUI_NORMAL)

    cat_colors = {'corner-top': (255, 50, 50),
                  'corner-bottom': (50, 50, 255),
                  'intersection-top': (255, 255, 50),
                  'intersection-side': (255, 50, 255),
                  'edge-top': (50, 255, 255),
                  'edge-size': (255, 100, 50),
                  'default': (255, 255, 255),
                  }

    for file in labels:  # dict of labelled_files

        img = cv.imread(input_folder + os.sep + file)
        img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # reversed indices, OK

        for category in labels[file]:  # dict of categories

            if not os.path.isdir(output_folder + os.sep + category):
                os.makedirs(output_folder + os.sep + category, exist_ok=True)

            # for label_pos in labels[file][category]:  # list of labels
            #     label_pos_scaled = int(int(label_pos[0]) * scale), int(int(label_pos[1]) * scale)
            #     draw_cross(img, label_pos_scaled, color=cat_colors[category])

            for label_pos in labels[file][category]:  # list of labels
                label_pos_scaled = int(int(label_pos[0]) * scale), int(int(label_pos[1]) * scale)
                # generate region proposals
                region = crop_out(img, random_offset(label_pos_scaled))

                # save image
                region_path = output_folder + os.sep + category + os.sep  # per category folder
                region_filename = file.split('.')[0] + '_(' + str(label_pos[0]) + ',' + str(label_pos[1]) + ').jpg'
                # print(region_path)

                tmp = cv.imwrite(region_path + region_filename, region)
                if not tmp:
                    print('!', end='')

                # cv.imshow(':-)', region)
                #
                # k = cv.waitKey(0)
                # if k == ord("q"):
                #     break

    # cv.destroyAllWindows()
