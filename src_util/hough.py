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
import skimage
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from labels import load_labels_pandas


def delaunay(points):
    tri = Delaunay(points)
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    for j, p in enumerate(points):
        plt.text(p[0] - 0.03, p[1] + 0.03, j, ha='right')
    for j, s in enumerate(tri.simplices):
        p = points[s].mean(axis=0)
        plt.text(p[0], p[1], '#%d' % j, ha='center')
    # plt.xlim(-0.5, 1.5)
    # plt.ylim(-0.5, 1.5)
    plt.show()


def connect_closer_than(points, canvas_shape, dist_max=100, alpha=100):
    dists = cdist(points, points)
    canvas = np.zeros(shape=canvas_shape)
    for i in range(dists.shape[0]):
        for j in range(i + 1, dists.shape[1]):
            if dists[i, j] <= dist_max:
                # print('+')
                d = skimage.draw.line(*points[i], *points[j])
                # print(d)
                canvas[d] += alpha

    canvas = canvas.clip(0, 255).astype(np.uint8)
    return canvas


def double_voronoi(points):
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor)
    plt.show()

    # https://stackoverflow.com/questions/36985185/fast-fuse-of-close-points-in-a-numpy-2d-vectorized
    for r in range(2, 20, 2):
        v_points = np.array(vor.vertices)
        tree = cKDTree(v_points)
        pairs_to_fuse = tree.query_pairs(r=r)

        # print(repr(rows_to_fuse))
        # print(repr(v_points[list(rows_to_fuse)]))

        # TODO results not disjoint
        v_points_close = np.vstack([v_points[a] + v_points[b] / 2 for a, b in pairs_to_fuse])

        points_close_indices = np.unique(np.array(list(pairs_to_fuse)).flatten())

        mask = np.ones(len(v_points), dtype=np.bool)
        mask[points_close_indices] = 0
        v_points_far = v_points[mask]

        v_points_fused = np.vstack([v_points_close, v_points_far])
        # plt.scatter(*v_points_fused.T)
        # plt.scatter(*v_points_close.T)
        # plt.scatter(*v_points_far.T)
        plt.xlim(-0.5, 1000.5)
        plt.ylim(-0.5, 1000.5)
        plt.title(f'{r=}')
        plt.show()


def hough_lines_point_set(points_cv):
    points_cv_reshaped = points_cv.reshape(-1, 1, 2)

    r = cv.HoughLinesPointSet(points_cv_reshaped, 1500, 2,
                              0, 1000, 2,  # rho
                              0, np.pi, np.pi / 180)  # theta

    # cv.HoughLinesPointSet(_point, lines_max, threshold,
    #                       min_rho, max_rho, rho_step,
    #                       min_theta, max_theta, theta_step[,_lines]    )

    plt.scatter(*r.T)
    plt.show()
    rpts = r.reshape((1, -1, 2))
    # xfm_pts = cv.perspectiveTransform(rpts, H).reshape((-1, 2))  # what's H

    # cv.utils.dumpInputArray(points_cv)  # was of no use


def connect_overlay(points_cv, img_rgb, th):
    net = connect_closer_than(points_cv, (img_rgb.shape[0], img_rgb.shape[1]), dist_max=th, alpha=255)
    # img_plus_net = np.clip(img_rgb // 4 + net[:, :, None], 0, 255).astype(np.uint8)
    img_plus_net = net
    plt.imshow(img_plus_net)
    plt.title(f'{th=}')
    # plt.axis(False)
    plt.show()


if __name__ == '__main__':
    """ Load points """
    input_folder = 'sirky_val'
    labels = load_labels_pandas(f'{input_folder}/labels.csv', use_full_path=False, keep_bg=False)

    scale = 1

    labels.x = (labels.x * scale).astype(np.intc)
    labels.y = (labels.y * scale).astype(np.intc)

    # for file in labels.image.unique():
    file = next(iter(labels.image.unique()[::-1]))
    cat = 'edge-side'
    file_labels = labels[labels.image == file]
    cat_labels = file_labels[file_labels.category == cat]

    img = cv.imread(input_folder + os.sep + file)
    img = cv.resize(img, None, fx=scale, fy=scale)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    canvas_cv = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # canvas_cv[cat_labels.y, cat_labels.x] = 255
    canvas_cv[file_labels.y, file_labels.x] = 255

    points = np.array([file_labels.x, file_labels.y]).T
    points_cv = np.array([file_labels.y, file_labels.x]).T

    """ Work with points """
    if True:

        """ Show original label points """
        plt.scatter(*points.T)
        # coord origin is top-left
        plt.gca().invert_yaxis()
        plt.show()

        """ Connect point to its closest """
        # for th in range(40, 250, 10):
        th = 800
        connect_overlay(points_cv, img_rgb, th * scale)

    if False:
        # double_voronoi(points)
        # delaunay(points)
        # c_img = np.dstack([canvas] * 3)

        line_points = np.linspace(200, 800, num=50)

    """ Show with CV """
    if False:
        window_name = 'hough'
        cv.namedWindow(window_name, cv.WINDOW_GUI_NORMAL)

        img_copy = img.copy()
        # for threshold in [2]:  # range(10, 200, 10):
        for maxLineGap in range(100, 2000, 100):

            lines = cv.HoughLinesP(canvas_cv, rho=2, theta=np.pi / 180, threshold=1,
                                   lines=None, minLineLength=None, maxLineGap=None)

            # def HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)

            if lines is not None:
                print(len(lines))
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # BGR

                cv.imshow(window_name, img)

                k = cv.waitKey(0)
                print('.')
                if k == ord("q"):
                    break
                img = img_copy.copy()
            else:
                print(':', end='')

        cv.destroyAllWindows()

    if False:
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
