import cv2 as cv
import numpy as np
import csv
from collections import defaultdict
import pathlib
import os
from labels import load_labels, img_dims
from random import randint


# Dead development branch - if you want to copy code, look into 'image_regions.py'


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)


# different from image_regions -- investigate
def draw_cross(img, center_pos, line_length=20, color=(255, 0, 0), width=8):
    global scale
    width = int(width * scale)
    line_length = int(line_length * scale)

    x, y = int(int(center_pos[0]) * scale), int(int(center_pos[1]) * scale)
    cv.line(img, (x - line_length, y - line_length), (x + line_length, y + line_length), color, width)
    cv.line(img, (x + line_length, y - line_length), (x - line_length, y + line_length), color, width)


if __name__ == '__main__':

    show_images = False
    save_images = not show_images

    scale = 1  # 0.25

    input_folder = 'sirky'
    output_folder = 'labelled_images_half'
    labels_file = 'labels.csv'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    labels = load_labels(input_folder + os.sep + labels_file, use_full_path=False)

    if show_images:
        cv.namedWindow(':)', cv.WINDOW_GUI_NORMAL)

    cat_colors = {
        'corner-top': (200, 50, 50),
        'corner-bottom': (50, 50, 200),
        'intersection-top': (200, 200, 50),
        'intersection-side': (200, 50, 200),
        'edge-top': (50, 255, 255),
        'edge-side': (255, 100, 50),
        'edge-bottom': (50, 255, 50),
        'default': (150, 150, 150),
    }

    for file in labels:  # dict of labelled_files
        img = cv.imread(input_folder + os.sep + file)
        img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # reversed indices, OK

        dims = img_dims[file]

        # draw labels
        for category in labels[file]:  # dict of categories
            for label_pos in labels[file][category]:  # list of labels
                draw_cross(img, label_pos, color=cat_colors[category])

        if save_images:
            path = output_folder + os.sep + file
            tmp = cv.imwrite(path, img)
            if not tmp:
                print(path)

        if show_images:
            cv.putText(img, file, (img.shape[1] // 5 * 4, img.shape[0] // 12 * 11),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                       5, cv.LINE_AA)
            cv.imshow(':)', img)

            k = cv.waitKey(0)
            if k == ord("q"):
                break

    if show_images:
        cv.destroyAllWindows()
