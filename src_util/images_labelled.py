# stdlib
import argparse
import csv
import os
import sys

# external
import cv2 as cv
import numpy as np

# local
from labels import load_labels_dict

"""
Show annotated images

Controls:
    A:  <-
        previous
           
    D:  -> 
        next
    
    Q:  quit
"""
window_name = 'Q=Quit, A=Prev, D=Next'


def draw_cross(img, center_pos, line_length=20, color=(0, 0, 0), width=6, scale=1):
    width = int(width * scale)
    line_length = int(line_length * scale)

    x, y = int(int(center_pos[0]) * scale), int(int(center_pos[1]) * scale)
    cv.line(img, (x - line_length, y - line_length), (x + line_length, y + line_length), color, width)
    cv.line(img, (x + line_length, y - line_length), (x - line_length, y + line_length), color, width)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input dataset params
    parser.add_argument('--cutout_size', '-c', type=int, default=64,
                        help="cutout=sample size")

    parser.add_argument('--per_image_samples', '-p', type=int, default=100,
                        help="background sample count")

    parser.add_argument('--val', '-v', action='store_true',
                        help="work with validation data (instead of training)")

    parser.add_argument('--background', '-b', action='store_true',
                        help="include background samples")

    # program behavior
    parser.add_argument('--show', '-s', action='store_true', default=False,
                        help="show images instead of saving")

    """
    python src_util/images_labelled.py -b -c 64 -p 100
    python src_util/images_labelled.py -b -c 64 -p 100 -v
    python src_util/images_labelled.py -b -c 64 -p 500
    python src_util/images_labelled.py -b -c 64 -p 500 -v
    
    """

    args = parser.parse_args()

    run_config = f'{args.cutout_size}x_050s_{args.per_image_samples}bg' \
                 + '_KPonly' * (not args.background) + '_val' * args.val

    scale = 1  # 0.25
    image_input_folder = 'sirky' + '_val' * args.val

    output_folder = 'images_annotated' + os.sep + run_config

    labels_path = 'datasets' + os.sep + run_config + os.sep + \
                  'labels_with_bg.csv' if args.background else 'labels.csv'

    # labels_path = 'image_regions_64_050_bg500/labels_with_bg.csv'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if not os.path.isfile(labels_path):
        print('Could not find labels file', labels_path)
        sys.exit(1)

    if not os.path.isdir(image_input_folder):
        print('Could not find image folder', image_input_folder)
        sys.exit(1)

    labels = load_labels_dict(labels_path, use_full_path=False)

    cat_colors = {
        'corner-top': (200, 50, 50),
        'corner-bottom': (50, 50, 200),
        'intersection-top': (200, 200, 50),
        'intersection-side': (200, 50, 200),
        'edge-top': (50, 255, 255),
        'edge-side': (255, 100, 50),
        'edge-bottom': (50, 255, 50),
        'background': (220, 220, 220),
    }

    if not args.show:
        # save
        print(f'Saving to {output_folder}')
        for file in labels:
            img = cv.imread(image_input_folder + os.sep + file)
            if img is None:
                print('failed', image_input_folder + os.sep + file)
                continue

            img = cv.resize(img, None, fx=scale, fy=scale)

            # draw labels
            for category in labels[file]:  # dict of categories
                for label_pos in labels[file][category]:  # list of labels
                    draw_cross(img, label_pos, color=cat_colors[category])

            path = output_folder + os.sep + file
            tmp = cv.imwrite(path, img)
            if not tmp:
                print('Failed to save', path)
    else:
        # show (interactive)
        cv.namedWindow(window_name, cv.WINDOW_GUI_NORMAL)

        images_total = len(labels)
        labels_keys = list(labels)
        i = 0
        while True:
            file = labels_keys[i]

            img = cv.imread(image_input_folder + os.sep + file)
            img = cv.resize(img, None, fx=scale, fy=scale)

            # draw labels
            for category in labels[file]:  # dict of categories
                for label_pos in labels[file][category]:  # list of labels
                    draw_cross(img, label_pos, color=cat_colors[category])

            cv.putText(img, '{} {}/{}'.format(file, i + 1, images_total),
                       (img.shape[1] * 3 // 4, img.shape[0] * 11 // 12),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                       5, cv.LINE_AA)
            cv.imshow(window_name, img)

            k = cv.waitKey(0)
            if k == ord("a") and i > 0:
                i -= 1

            if k == ord("d") and i < images_total - 1:
                i += 1

            if k == ord("q") or i >= images_total - 1:
                break

        cv.destroyAllWindows()
