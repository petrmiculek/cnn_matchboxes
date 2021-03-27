import os

import cv2 as cv
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data

from labels import load_labels

if __name__ == '__main__':
    input_folder = 'sirky_val'
    labels = load_labels(f'{input_folder}/labels.csv')
    file = list(labels)[-1]
    fixed_cat = 'edge-side'

    cat_labels = np.array([label for label in labels[file][fixed_cat]], dtype=np.intc)

    file_labels = np.array([label
                   for cat in labels[file]
                   for label in labels[file][cat]])  # -> x,y

    img = cv.imread(file)

    canvas = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    canvas[cat_labels[:, 1], cat_labels[:, 0]] = 255
    canvas[file_labels[:, 1], file_labels[:, 0]] = 255

    # c_img = np.dstack([canvas] * 3)

    window_name = 'hough'
    cv.namedWindow(window_name, cv.WINDOW_GUI_NORMAL)
    img_copy = img.copy()
    for threshold in [2]:  # range(10, 200, 10):
        for maxLineGap in range(0, 4001, 1000):

            lines = cv.HoughLinesP(canvas, 1, np.pi / 180, threshold, lines=None, minLineLength=3, maxLineGap=maxLineGap)
            # def HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)

            if lines is not None:
                print(len(lines))
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

                cv.imshow(window_name, img)

                k = cv.waitKey(0)
                print('.')
                # if k == ord("q"):
                img = img_copy.copy()
            else:
                print(':', end='')

    cv.destroyAllWindows()

    """
    # Image.fromarray(img).show()
    theta = np.pi / 180
    for rho in [1, 2]:
        for threshold in [2]:  # range(10, 200, 10):
            for minLineLength in [80]:  # range(10, 200, 10):
                for maxLineGap in range(0, 4000, 200):
                    lines = cv.HoughLinesP(canvas, rho, theta, threshold, lines=3, minLineLength=minLineLength, maxLineGap=maxLineGap)
                    print(f'{rho} {theta} {threshold} {minLineLength} {maxLineGap} -> {len(lines) if lines is not None else 0}')
    """

    """
    # just try scikit-image pHough
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(canvas, theta=tested_angles)

    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(canvas, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]

    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(canvas, cmap=cm.gray)
    ax[2].set_ylim((canvas.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    plt.tight_layout()
    plt.show()
    """
