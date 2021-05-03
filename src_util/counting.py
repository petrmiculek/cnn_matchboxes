# stdlib
import os

# external
import sys

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line as draw_line
import skimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, cosine

# local
from eval_images import crop_to_prediction, load_image
from labels import load_labels, resize_labels


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


def connect_closer_than(points, canvas_shape, dist_max=100, alpha=100):
    dists = cdist(points, points)
    canvas = np.zeros(shape=canvas_shape)
    for i in range(dists.shape[0]):
        for j in range(i + 1, dists.shape[1]):
            if dists[i, j] <= dist_max:
                d = draw_line(*points[i], *points[j])
                canvas[d] += alpha

    canvas = canvas.clip(0, 255).astype(np.uint8)
    return canvas


def connect_overlay(points_cv, img_rgb, th):
    net = connect_closer_than(points_cv, (img_rgb.shape[0], img_rgb.shape[1]), dist_max=th, alpha=255)
    # img_plus_net = np.clip(img_rgb // 4 + net[:, :, None], 0, 255).astype(np.uint8)
    img_plus_net = net
    plt.imshow(img_plus_net)
    plt.title('th=', th)
    # plt.axis(False)
    plt.show()


def get_gt_points(file_labels, cat=None, cv=False):
    lines = file_labels
    if cat is not None:
        if isinstance(cat, str):
            lines = file_labels[file_labels.category == cat]
        else:  # assume list
            lines = file_labels[file_labels.category.isin(cat)]

    if len(lines) == 0:
        return np.array([])

    if cv:
        # openCV order
        return np.array([lines.y, lines.x]).T
    else:
        # numpy, scikit order
        return np.array([lines.x, lines.y]).T


def inverse_index(arr, index):
    mask = np.ones(len(arr), dtype=np.bool)
    mask[index] = 0
    return arr[mask]  # could use a copy


def connect_points(canvas, pts1, pts2, c=200):
    if len(pts1) != len(pts2):
        print('warning, len mismatch 1: {} x 2: {}'.format(len(pts1), len(pts2)))

    for i in range(len(pts1)):
        d = draw_line(*pts1[i], *pts2[i])
        canvas[d] = c


def axis_direction(pts_from, pts_to):
    """Axis direction as a unit vector

    input points must be aligned already
    """
    direction = (pts_to - pts_from).mean(axis=0)
    return direction / (np.sqrt(np.sum(direction ** 2)) + np.finfo(np.float64).eps)


def closest_pairs_greedy(pts_from, pts_to, indices_only=False):
    """Get pts_to ordered by greedy closest-pairing

    greedy = choose the closest pair, remove the pair from the pool
    pairs are inherently disjoint


    """
    assert pts_to.ndim == pts_from.ndim == 2
    assert pts_to.shape[1] == pts_to.shape[1] == 2
    assert len(pts_to) >= len(pts_from), "not enough points to connect to"

    dists = cdist(pts_from, pts_to)
    # assert len(dists.shape) == 2
    fill_value = np.max(dists) + 1
    if indices_only:
        ret = np.zeros((pts_from.shape[0]))
    else:
        ret = np.zeros_like(pts_from)

    for _ in range(len(pts_from)):
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        # idx = [from, to]
        if indices_only:
            ret[idx[0]] = idx[1]
        else:
            ret[idx[0]] = pts_to[idx[1]]

        dists[:, idx[1]] = fill_value
        dists[idx[0], :] = fill_value

    return ret


def closest_pairs_in_order(pts_from, pts_to, indices_only=False):
    """ Find closest point_from for each point_to"""

    assert pts_to.ndim == pts_from.ndim == 2
    assert pts_to.shape[1] == pts_to.shape[1] == 2
    assert len(pts_to) >= len(pts_from), "not enough points to connect to"

    dists = cdist(pts_from, pts_to)
    fill_value = np.max(dists) + 1
    ret = []
    for to_pts_to in dists:
        idx = np.argmin(to_pts_to)

        if indices_only:
            ret.append(idx)
        else:
            ret.append(pts_to[idx])

        dists[:, idx] = fill_value

    return np.array(ret)


def count_points_on_line(points, line, dst_th=10, show=False):
    """Count `points` on `line` """
    if len(points) == 0:
        return 0  # just corners

    p1, p2 = line
    line_len = np.linalg.norm(p2 - p1)

    tolerance = [10, 10]
    low = np.min(line, axis=0) - tolerance
    high = np.max(line, axis=0) + tolerance
    idx_inside = np.all(np.logical_and(points >= low, points <= high), axis=1)
    wrapper = points[idx_inside]  # obálka = rectangle aligned with axes

    # distance(points, line)
    dists = np.abs(np.cross(p2 - p1, p1 - wrapper))
    if line_len != 0:
        # print(line_len)
        dists = dists / line_len
    else:
        print(f'count_points_on_line: zero-length line: {p1} -> {p2}', file=sys.stderr)

    # thresholding
    on_the_line = np.argwhere(dists < dst_th)[:, 0]

    if show:
        plt.scatter(*points.T, marker='o')
        plt.scatter(*wrapper[on_the_line].T, marker='x')
        plt.scatter(*line.T, marker='+')
        
        plt.gca().invert_yaxis()  # if new plot, axis needs to be inverted
        # (disable ^^^ when using imshow)
        plt.show()

    return len(on_the_line)


def consensus(arr):
    count, votes = np.unique(arr, return_counts=True)
    count_voted = count[votes == votes.max()].mean()
    return count_voted


def get_top_back_point(pts_bottom, pts_top):
    if len(pts_top) != 4 and len(pts_bottom) != 3:
        raise UserWarning('Triangle center method: invalid number of points bot:{}, top:{}'
                          .format(len(pts_bottom), len(pts_top)))
    bot_mean = pts_bottom.mean(axis=0)
    top_sum = pts_top.sum(axis=0)
    scores = []

    for i, pt_top_back in enumerate(pts_top):
        # center of the remaining three front points
        top_mean = (top_sum - pt_top_back) / 3

        dist_to_top_front = np.linalg.norm(bot_mean - top_mean)
        dist_to_top_back = np.linalg.norm(bot_mean - pt_top_back)

        angle_factor = (1 - cosine(bot_mean - pt_top_back, bot_mean - top_mean))
        distance_factor = dist_to_top_back - dist_to_top_front
        score = angle_factor * distance_factor
        scores.append(score)

    return np.argmax(scores)


def ordered_along_line(points, line_end, line_start=np.array([0, 0])):
    side_size2 = ((line_start - line_end) ** 2).sum()  # L2-squared
    if side_size2 == 0:
        raise ValueError('Line has zero length')  # ZeroDivisionError

    ts = []
    for pt in points:
        t = ((pt - line_start) * (line_end - line_start)).sum() / side_size2
        ts.append(t)
        # p = 0 + t * (line_end - 0)  # projected point - unused

    ts = np.array(ts)
    ordering = np.argsort(ts)
    return ordering


def count_points_tlr(labels):
    """ 
    match front top points (3) with the corresponding bottom front points (3)

    count points on y-axis (up, bottom-top)

    connect corner points on xz-axes

        order of vertical lines easily obtained from the bottom->top direction

    """

    corners_top = get_gt_points(labels, 'corner-top', cv=False)
    corners_bottom = get_gt_points(labels, 'corner-bottom', cv=False)

    if not len(corners_bottom) == 3 or not len(corners_top) == 4:
        return -1

    farthest_top_idx = get_top_back_point(corners_bottom, corners_top)

    corners_top_front = inverse_index(corners_top, farthest_top_idx)  # s/matched/front
    # ^ using values, not indices => harder to find original indices -- do I need them?

    top_matched_indices, up_shift = match_top_to_bottom(corners_bottom, corners_top_front)

    corners_top_front = corners_top_front[top_matched_indices]

    # XZ-axes

    # perpendicular to `up_shift`
    side_direction = np.array([up_shift[1], up_shift[0] * -1])

    vertical_lines_means = np.dstack([corners_bottom, corners_top_front]).mean(axis=2)

    # order vertical lines along the horizontal direction (left to right)
    side_ordering = ordered_along_line(vertical_lines_means, side_direction)

    # if False:
    #     # verified, it works
    #     for i in side_ordering:
    #         plt.scatter(*pts_top[farthest_top_idx].T, marker='d')
    #         plt.scatter(*pts_bottom.T, marker='o')
    #         plt.scatter(*pts_top_front.T, marker='o')
    # 
    #         plt.scatter(*pts_bottom[i].T, marker='+')
    #         plt.scatter(*pts_top_matched[i].T, marker='x')
    #         plt.gca().invert_yaxis()
    #         plt.show()

    i0, i1, i2 = side_ordering
    # starting from:
    # left
    line_x0 = corners_bottom[[i0, i1]]
    line_x1 = corners_top_front[[i0, i1]]

    # front
    line_z0 = corners_bottom[[i1, i2]]
    line_z1 = corners_top_front[[i1, i2]]

    # right
    line_x2 = np.array([corners_top_front[i2], corners_top[farthest_top_idx]])

    # back
    line_z2 = np.array([corners_top[farthest_top_idx], corners_top_front[i0]])

    # X and Z axes can get switched up but it doesn't affect the counting

    # X-axis = left-right
    pts_x = get_gt_points(labels, cat=['edge-top', 'edge-bottom'], cv=False)
    counts_x = [
        count_points_on_line(pts_x, line_x0),
        count_points_on_line(pts_x, line_x1),
        count_points_on_line(pts_x, line_x2),
    ]
    count_x = consensus(counts_x)

    # Y-axis = top-bottom
    pts_y = get_gt_points(labels, cat='edge-side', cv=False)

    count_y = []
    for i in range(3):
        line_y = np.vstack([corners_bottom[i], corners_top_front[i]])
        c = count_points_on_line(pts_y, line_y)
        count_y.append(c)

    count_y = consensus(count_y)

    # Z-axis = front-back
    pts_z = get_gt_points(labels, cat=['edge-top', 'edge-bottom'], cv=False)
    counts_z = [
        count_points_on_line(pts_z, line_z0),
        count_points_on_line(pts_z, line_z1),
        count_points_on_line(pts_z, line_z2)
    ]
    count_z = consensus(counts_z)

    print((count_x, count_y, count_z), (len(pts_x), len(pts_y), len(pts_z)))

    count = (count_x + 1) * (count_y + 1) * (count_z + 1)
    # +2 corners per dim,
    # -1 because line edges = (vertices - 1)

    return count


def match_top_to_bottom(corners_bottom, corners_top_front):
    if len(corners_bottom) != len(corners_top_front):
        raise UserWarning(f'Top-bottom matching failed - different number of points '
                          f'{len(corners_bottom)} x {len(corners_top_front)}')
    # get axis shift vector
    up_shift = corners_top_front.mean(axis=0) - corners_bottom.mean(axis=0)

    # shift bottom to top
    pts_bottom_shifted = corners_bottom + up_shift

    # connect shifted bottom to closest top
    top_indices_matched_ordered = cdist(pts_bottom_shifted, corners_top_front).argmin(axis=1)

    # check unique indices
    idx_len = top_indices_matched_ordered.shape[0]
    expected_sum = (idx_len * (idx_len - 1) / 2)
    if top_indices_matched_ordered.sum() != expected_sum:
        print(top_indices_matched_ordered.sum(), expected_sum)
        raise UserWarning(f'Top-bottom matching failed - non-unique pairs: {top_indices_matched_ordered}')

    return top_indices_matched_ordered, up_shift


def main():
    """ Load GT points """
    input_folder = 'sirky'  # + '_val'
    labels_path = os.path.join(input_folder, 'labels.csv')
    labels = load_labels(labels_path, use_full_path=False, keep_bg=False)

    scale = 0.25
    center_crop_fraction = 1.0
    model_crop_delta = 63
    labels = resize_labels(labels, scale, model_crop_delta=model_crop_delta, center_crop_fraction=center_crop_fraction)

    counts_gt = pd.read_csv(os.path.join('sirky', 'count.txt'),  # todo fixed count gt folder
                            header=None,
                            names=['image', 'cnt'],
                            dtype={'image': str, 'cnt': np.int32})

    # file = next(iter(labels.image.unique()[::-1]))
    for file in labels.image.unique():
        # :params:
        # file => img_path, file_labels (gt-style expected)
        # config: scale, center_crop_fraction, model_crop_delta,
        # flags: cv,

        print(file)
        file_labels = labels[labels.image == file]
        img_path = input_folder + os.sep + file
        img, a = load_image(img_path, scale, center_crop_fraction)
        img, b = crop_to_prediction(img, (img.shape[0] - model_crop_delta, img.shape[1] - model_crop_delta))

        """ USING CV=FALSE """
        pts_top = get_gt_points(file_labels, 'corner-top', cv=False)
        pts_bottom = get_gt_points(file_labels, 'corner-bottom', cv=False)

        if False:
            plt.imshow(img)
            plt.plot(pts_bottom[:, 0], pts_bottom[:, 1], '+')
            plt.plot(pts_top[:, 0], pts_top[:, 1], 'o')
            # plt.plot(*pts_top[farthest_top_idx], 'x')
            plt.title(file)
            plt.show()
            print('')
            continue

        if len(pts_top) == 4 and len(pts_bottom) == 3:
            count_pred = count_points_tlr(file_labels)
            count_gt = np.array(counts_gt[counts_gt.image == file].cnt)[0]

            print(f'pred = {count_pred}, gt = {count_gt}')

        """ older attempts """
        if False:
            canvas_cv = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            points = get_cat_points(file_labels, cv=False)
            points_cv = get_cat_points(file_labels, cv=True)
            canvas_cv[points_cv[:, 0], points_cv[:, 1]] = 255

        if False:
            plt.imshow(img)
            plt.show()

        if False:
            hull = ConvexHull(pts_top)

            # plt.plot(pts_top[:, 0], pts_top[:, 1], 'o')
            for simplex in hull.simplices:
                plt.plot(*pts_top[simplex[0], :], '*')  # starting point - not unique
                plt.plot(pts_top[simplex, 0], pts_top[simplex, 1], 'k-')

            plt.show()

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
            # still CV
            theta = np.pi / 180
            for rho in [1, 2]:
                for threshold in [2]:  # range(10, 200, 10):
                    for minLineLength in [80]:  # range(10, 200, 10):
                        for maxLineGap in range(0, 4000, 200):
                            lines = cv.HoughLinesP(canvas_cv, rho, theta, threshold, lines=3,
                                                   minLineLength=minLineLength, maxLineGap=maxLineGap)
                            print(
                                f'{rho} {theta} {threshold} {minLineLength} {maxLineGap} -> {len(lines) if lines is not None else 0}')

        if False:
            # only now scikit-image
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
            h, theta, d = hough_line(canvas_cv, theta=tested_angles)

            # Generating figure 1
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            ax = axes.ravel()

            ax[0].imshow(canvas_cv, cmap=cm.gray)
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

            ax[2].imshow(canvas_cv, cmap=cm.gray)
            ax[2].set_ylim((canvas_cv.shape[0], 0))
            ax[2].set_axis_off()
            ax[2].set_title('Detected lines')

            for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
                (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
                ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

            plt.tight_layout()
            plt.show()

        if False:

            """ Use categories """
            # start from the top
            pts_top = get_cat_points(file_labels, 'corner-top', cv=True)
            pts_bottom = get_cat_points(file_labels, 'corner-bottom', cv=True)

            print(f'top: {len(pts_top)}, bottom: {len(pts_bottom)}')

            failed = False

            if len(pts_top) == 2:
                if len(pts_bottom) == 2:
                    print('side view')
                else:
                    print('failed')
                    failed = True

            elif len(pts_top) == 3:
                if len(pts_bottom) == 3:
                    print('side-oblique view')
                else:
                    print('failed')
                    failed = True

            elif len(pts_top) == 4:
                if len(pts_bottom) == 0:
                    print('top view')
                elif len(pts_bottom) == 3:
                    print('oblique (corner) view')

                elif len(pts_bottom) == 2:
                    print('side-top view')
                else:
                    print('failed')
                    failed = True

            else:
                print('view recognition failed')
                failed = True

            if failed:
                plt.imshow(img)
                plt.title(f'top: {len(pts_top)}, bottom: {len(pts_bottom)}')
                plt.show()

            continue

            hull = ConvexHull(pts_top)

            plt.plot(pts_top[:, 0], pts_top[:, 1], 'o')
            for simplex in hull.simplices:
                plt.plot(pts_top[:, 0], pts_top[:, 1], 'o')
                plt.plot(*pts_top[simplex[0], :], '*')
                plt.plot(pts_top[simplex, 0], pts_top[simplex, 1], 'k-')
                plt.show()

            plt.show()

        if False:
            """
            This started from the bottom.
            
            Using CV order for everything
            1) connect closest bottom-top pairs
                start with bottom points
                connect each with its closest top point
                
            2) connect top front-to-back
                start with front-top point found in previous step
                connect it to its closest top point 
                
            3) connect everything left-to-right
            """
            pts_top = get_cat_points(file_labels, 'corner-top', cv=True)
            pts_bottom = get_cat_points(file_labels, 'corner-bottom', cv=True)

            # reorder to descending Y => from lowest to highest (image-coords start in top-left)
            pts_bottom = pts_bottom[np.argsort(pts_bottom[:, 0])[::-1]]

            if len(pts_bottom) == 3:
                # consider any two of the bottom points to be 'front'
                pts_bottom_front = pts_bottom[:2]
            else:
                pts_bottom_front = pts_bottom

            if False:
                dists_bottom_top = cdist(pts_top, pts_bottom_front)
                closest_top_front_indices = np.argmin(dists_bottom_top, axis=0)
                assert len(np.unique(closest_top_front_indices)) == len(closest_top_front_indices), \
                    'bottom-top closest pairs not disjoint'
                # ordering for bottom-to-top-front
                pts_top_front = pts_top[closest_top_front_indices]
                # ^ see doc for shortcomings
            else:
                closest_top_front_indices = closest_pairs_in_order(pts_bottom_front, pts_top, indices_only=True)
                pts_top_front = pts_top[closest_top_front_indices]

            # top non-front points
            pts_top_back = inverse_index(pts_top, closest_top_front_indices)  # compare

            if len(pts_top_front) != len(pts_top_front):
                print('warning, top len - front: {} x back: {}'.format(len(pts_top_front), len(pts_top_back)))

            pts_top_back = closest_pairs_greedy(pts_top_front, pts_top_back)

            # connect horizontally (X)
            pts_left = np.vstack([pts_top_front[0], pts_top_back[0], pts_bottom_front[0]])
            pts_right = np.vstack([pts_top_front[1], pts_top_back[1], pts_bottom_front[1]])

            if False:
                for i in range(len(pts_bottom_front)):
                    pt = pts_bottom_front[i]
                    ccl = skimage.draw.disk(pt, radius=5 + i * 10)
                    canvas_cv[ccl] = 100

            # Z
            # bottom -> top line
            connect_points(canvas_cv, pts_bottom_front, pts_top_front, c=120)

            # Y
            # top-front -> top-back line
            connect_points(canvas_cv, pts_top_front, pts_top_back, c=120)

            # X
            # left -> right
            connect_points(canvas_cv, pts_left, pts_right, c=120)

            # connect_points(canvas_cv, pts_top_front[::2], pts_top_front[1::2])
            # connect_points(canvas_cv, pts_top_back[::2], pts_top_back[1::2])
            # connect_points(canvas_cv, pts_bottom_front[::2], pts_bottom_front[1::2])

            # x = []  # left-to-right
            # y = []  # front-to-back
            # z = []  # top-to-bottom

            y = axis_direction(pts_top_front, pts_top_back)
            z = axis_direction(pts_bottom_front, pts_top_front)
            x = axis_direction(pts_left, pts_right)

            # print(f'Axes in cv order[y, x]\n{x=}\n{y=}\n{z=}\n')

            plt.imshow(canvas_cv)
            plt.title(file)
            plt.show()

            pts_left_top = np.vstack([pts_top_front[0], pts_top_back[0]])
            pts_right_top = np.vstack([pts_top_front[1], pts_top_back[1]])

            pts_left_front = np.vstack([pts_bottom_front[0], pts_top_front[0]])
            pts_right_front = np.vstack([pts_bottom_front[1], pts_top_front[1]])

            pts_x = get_cat_points(file_labels, cat=['edge-top', 'edge-bottom'], cv=True)
            counts_x = [
                count_points_on_line(pts_x, pts_top_front),
                count_points_on_line(pts_x, pts_top_back),
                count_points_on_line(pts_x, pts_bottom_front),
            ]
            count_x = consensus(counts_x)

            pts_y = get_cat_points(file_labels, cat='edge-top', cv=True)
            counts_y = [
                count_points_on_line(pts_y, pts_left_top),
                count_points_on_line(pts_y, pts_right_top)
            ]
            count_y = consensus(counts_y)

            pts_z = get_cat_points(file_labels, cat='edge-side', cv=True)
            counts_z = [
                count_points_on_line(pts_z, pts_left_front),
                count_points_on_line(pts_z, pts_right_front)
            ]
            count_z = consensus(counts_z)

            # todo
            # counting points needs to use also intersection categories
            # instead of all points, also just pass categories accordingly

            pred_total = (count_x + 1) * (count_y + 1) * (count_z + 1)
            # +2 corners per dim,
            # -1 because line edges = (vertices - 1)

            print('pred =', pred_total)
            count_gt = np.array(counts_gt[counts_gt.image == file].cnt)[0]
            if count_gt != pred_total:
                print('wrong, gt =', count_gt)

            print('...')
            tmp = input()
            if tmp == 'q':
                raise ValueError()

        def connect_closest_fails():
            #  discovered on 20201020_121235.jpg
            plt.scatter(*points_cv.T, marker='+')
            plt.scatter(*pts_top_front.T, marker='o')
            plt.scatter(*pts_top_back.T, marker='o')
            plt.title('"Connest closest" assertion breaks')
            plt.show()
        
            # solution
            # A) find single closest pair
            #    A.A) get more points using the same direction -- different angles
            #    A.B) remove this point and search again
        
            # B) get direction and for each point find one closest to that line
            #       -- angles may differ


# todo uncomment
if __name__ == '__main__':
    main()
