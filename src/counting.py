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
from general import inverse_indexing
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

        # plt.gca().invert_yaxis()  # if new plot, axis needs to be inverted
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


def count_points_l(file_labels):
    """

    Both top and bottom lines are made to face right

    :param file_labels:
    :return:
    """
    corners_top = get_gt_points(file_labels, 'corner-top', cv=False)
    corners_bottom = get_gt_points(file_labels, 'corner-bottom', cv=False)
    if not len(corners_top) == 2 or not len(corners_bottom) == 2:
        return -1

    bottom_left, bottom_right = 0, 1
    line_top = corners_bottom[bottom_right] - corners_bottom[bottom_left]
    if np.any(line_top != to_right_quadrants(line_top)):
        bottom_left, bottom_right = bottom_right, bottom_left

    top_left, top_right = 0, 1
    line_bottom = corners_top[top_right] - corners_top[top_left]
    if np.any(line_bottom != to_right_quadrants(line_bottom)):
        top_left, top_right = top_right, top_left

    lines_x = [
        corners_bottom[[bottom_left, bottom_right]],
        corners_top[[top_left, top_right]]
    ]

    lines_y = [
        np.c_[corners_bottom[bottom_left], corners_top[top_left]].T,
        np.c_[corners_bottom[bottom_right], corners_top[top_right]].T
    ]

    lines_z = None

    count = final_count(file_labels, lines_x, lines_y, lines_z)

    return count


def count_points_t(file_labels):
    corners_top = get_gt_points(file_labels, 'corner-top', cv=False)
    if not len(corners_top) == 4:
        return -1

    rect_ordering = ConvexHull(corners_top)

    i, j = rect_ordering.simplices[0]

    line1 = corners_top[j] - corners_top[i]

    parallel_indices = find_parallel(corners_top, line1)  # redundant work

    i, j, m, n = parallel_indices

    lines_x = [
        corners_top[[i, j]],
        corners_top[[m, n]]
    ]

    lines_y = None

    lines_z = [
        corners_top[[i, m]],
        corners_top[[j, n]]
    ]

    count = final_count(file_labels, lines_x, lines_y, lines_z)

    return count


def count_points_tl(file_labels):
    corners_top = get_gt_points(file_labels, 'corner-top', cv=False)
    corners_bottom = get_gt_points(file_labels, 'corner-bottom', cv=False)

    if not len(corners_bottom) == 2 or not len(corners_top) == 4:
        return -1

    # get horizontal direction from bottom points
    horizontal = corners_bottom[1] - corners_bottom[0]

    parallels_indices = find_parallel(corners_top, horizontal)

    corners_top_a = corners_top[parallels_indices[[0, 1]]]
    corners_top_b = corners_top[parallels_indices[[2, 3]]]

    if False:
        # checking parallel lines
        plt.imshow(img)
        plt.title(file)
        plt.scatter(*corners_bottom.T, marker='d')
        plt.scatter(*corners_top.T, marker='o')
        plt.scatter(*corners_top_a.T, marker='+')
        plt.scatter(*corners_top_b.T, marker='x')
        plt.show()

    vertical = np.array([horizontal[1], horizontal[0] * -1])

    lines = [corners_bottom, corners_top_a, corners_top_b]

    means = [corners_bottom.mean(axis=0),
             corners_top_a.mean(axis=0),
             corners_top_b.mean(axis=0)]

    vertical_ordering = ordered_along_line(means, vertical)

    if vertical_ordering[2] == 0:
        # bottom is either at the beginning or the end
        # make sure it is at the beginning
        vertical_ordering = vertical_ordering[::-1]

    corners_top_front = lines[vertical_ordering[1]]
    corners_top_back = lines[vertical_ordering[2]]

    if False:
        # checking ordering of top rows
        plt.imshow(img)
        plt.title(file)
        plt.scatter(*corners_bottom.T, marker='d')
        plt.scatter(*corners_top_front.T, marker='+')
        plt.scatter(*corners_top_back.T, marker='x')
        plt.show()

    lines_x = [
        corners_bottom[[0, 1]],
        corners_top_front[[0, 1]],
        corners_top_back[[0, 1]]
    ]

    lines_y = [
        np.vstack([corners_bottom[0], corners_top_front[0]]),
        np.vstack([corners_bottom[1], corners_top_front[1]])
    ]

    lines_z = [
        np.array([corners_top_front[0], corners_top_back[0]]),
        np.array([corners_top_front[1], corners_top_back[1]])
    ]

    count = final_count(file_labels, lines_x, lines_y, lines_z)

    return count


def find_parallel(points, direction):
    """
    Intended for finding 2 parallel lines between 4 points

    :param points:
    :param direction:
    :return:
    """
    # create lines from points so that they have the highest sum of cosine similarities
    if len(points) != 4:
        raise NotImplementedError(f'Finding parallel lines fails for {len(points)} != 4 points')

    direction = direction / np.linalg.norm(direction)
    best_score = np.inf
    best_indices = [-1, -1, -1, -1]  # from-to, from-to

    for i, _ in enumerate(points):
        for j, _ in enumerate(points):
            if i == j:
                continue

            line1 = points[j] - points[i]
            m, n = inverse_indexing(np.arange(len(points)), [i, j])
            line1 = line1 / np.linalg.norm(line1)

            for line2_reversed in [False, True]:
                if line2_reversed:
                    m, n = n, m

                line2 = points[n] - points[m]

                # normalize to unit vector
                line2 = line2 / np.linalg.norm(line2)

                # # dot + arccos => angle
                # di = np.dot(line1, line2)
                # d1 = np.dot(line1, direction)
                # d2 = np.dot(line2, direction)
                #
                # ai = np.arccos(di)
                # a1 = np.arccos(d1)
                # a2 = np.arccos(d2)
                #
                # err1 = np.abs(a1 - ai)
                # err2 = np.abs(a2 - ai)

                err1 = cosine(direction, line1)
                err2 = cosine(direction, line2)
                err_inter = cosine(line1, line2)
                curr_score = err1 + err2 + err_inter

                # print([i, j, m, n], err1, err2, err_inter, (curr_score))

                if curr_score < best_score:
                    best_score = curr_score
                    best_indices = [i, j, m, n]
                    # print('^')

    return np.array(best_indices)


def count_points_lr(file_labels):
    """

    match top points (3) with the corresponding bottom points (3)

    order vertical lines based on their horizontal position (left-front-right)

    connect corner points on xz-axes (horizontal lines)

    count points on XYZ-axes

    :param file_labels:
    :return:
    """
    corners_top = get_gt_points(file_labels, 'corner-top', cv=False)
    corners_bottom = get_gt_points(file_labels, 'corner-bottom', cv=False)

    if not len(corners_bottom) == 3 or not len(corners_top) == 3:
        return -1

    top_matched_indices, up_shift = match_top_to_bottom(corners_bottom, corners_top)
    corners_top = corners_top[top_matched_indices]

    side_direction = np.array([up_shift[1], up_shift[0] * -1])

    vertical_lines_means = np.dstack([corners_bottom, corners_top]).mean(axis=2)

    # order vertical lines along the horizontal direction (left to right)
    side_ordering = ordered_along_line(vertical_lines_means, side_direction)

    i_left, i_front, i_right = side_ordering

    lines_x = [
        corners_bottom[[i_left, i_front]],
        corners_top[[i_left, i_front]],
    ]

    lines_y = []
    for i in range(3):
        line_y = np.vstack([corners_bottom[i], corners_top[i]])
        lines_y.append(line_y)

    lines_z = [
        corners_bottom[[i_front, i_right]],
        corners_top[[i_front, i_right]],
    ]

    count = final_count(file_labels, lines_x, lines_y, lines_z)

    return count


def count_points_tlr(file_labels, quiet=True):
    """Count points in a 3-side view

    split top points into front (3) and back (1)

    match front top points (3) with the corresponding bottom front points (3)

    order vertical lines based on their horizontal position (left-front-right)

    connect corner points on xz-axes (horizontal lines)

    count points on XYZ-axes

    """

    corners_top = get_gt_points(file_labels, 'corner-top', cv=False)
    corners_bottom = get_gt_points(file_labels, 'corner-bottom', cv=False)

    if not len(corners_bottom) == 3 or not len(corners_top) == 4:
        return -1

    farthest_top_idx = get_top_back_point(corners_bottom, corners_top)

    corners_top_front = inverse_indexing(corners_top, farthest_top_idx)
    # ^ using values, not indices => harder to find original indices -- do I need them?

    top_matched_indices, up_shift = match_top_to_bottom(corners_bottom, corners_top_front)

    corners_top_front = corners_top_front[top_matched_indices]

    # perpendicular to `up_shift`
    side_direction = np.array([up_shift[1], up_shift[0] * -1])

    vertical_lines_means = np.dstack([corners_bottom, corners_top_front]).mean(axis=2)

    # order vertical lines along the horizontal direction (left to right)
    horizontal_ordering = ordered_along_line(vertical_lines_means, side_direction)

    """
    if False:
    # checking front-back decision
    for i in side_ordering:
        plt.scatter(*pts_top[farthest_top_idx].T, marker='d')
        plt.scatter(*pts_bottom.T, marker='o')
        plt.scatter(*pts_top_front.T, marker='o')

        plt.scatter(*pts_bottom[i].T, marker='+')
        plt.scatter(*pts_top_matched[i].T, marker='x')
        plt.gca().invert_yaxis()
        plt.show()
    """

    i_left, i_front, i_right = horizontal_ordering

    lines_x = [
        corners_bottom[[i_left, i_front]],
        corners_top_front[[i_left, i_front]],
        np.array([corners_top_front[i_right], corners_top[farthest_top_idx]])
    ]

    lines_y = []
    for i in range(len(corners_bottom)):
        line_y = np.vstack([corners_bottom[i], corners_top_front[i]])
        lines_y.append(line_y)

    lines_z = [
        corners_bottom[[i_front, i_right]],
        corners_top_front[[i_front, i_right]],
        np.array([corners_top[farthest_top_idx], corners_top_front[i_left]])
    ]

    # X and Z axes can get switched up but it doesn't affect the counting

    count = final_count(file_labels, lines_x, lines_y, lines_z)

    return count


def final_count(labels, lines_x=None, lines_y=None, lines_z=None, show=False, cv=False):
    # X-axis = left-right
    if lines_x is not None:
        pts_x = get_gt_points(labels, cat=['edge-top', 'edge-bottom'], cv=cv)
        count_x = count_points_lines_all(lines_x, pts_x, show=show)
    else:
        count_x = 0

    # Y-axis = top-bottom
    if lines_y is not None:
        pts_y = get_gt_points(labels, cat='edge-side', cv=cv)
        count_y = count_points_lines_all(lines_y, pts_y, show=show)
    else:
        count_y = 0
    # Z-axis = front-back
    if lines_z is not None:
        pts_z = get_gt_points(labels, cat=['edge-top', 'edge-bottom'], cv=cv)
        count_z = count_points_lines_all(lines_z, pts_z, show=show)
    else:
        count_z = 0

    # print((count_x, count_y, count_z), (len(pts_x), len(pts_y), len(pts_z)))

    count = (count_x + 1) * (count_y + 1) * (count_z + 1)
    # +2 corners per dim,
    # -1 because line edges = (vertices - 1)

    return count


def to_right_quadrants(vector):
    """Flip vector if pointing left

    quadrants:

      II |  I
    -----+-----
     III | IV


    "stay (x-)positive"

    :param vector:
    :return:
    """
    if vector[0] == 0:
        vector[1] = np.abs(vector[1])

    elif vector[0] < 0:
        vector = -vector

    return vector


def count_points_lines_all(lines, points, show=False):
    counts = []
    for curr_line in lines:
        c = count_points_on_line(points, curr_line, show=show)
        counts.append(c)
    count_x = consensus(counts)
    return count_x


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


def count_crates(keypoints, quiet=True):
    corners_top = get_gt_points(keypoints, 'corner-top', cv=False)
    corners_bottom = get_gt_points(keypoints, 'corner-bottom', cv=False)
    count_pred = -1
    try:
        if len(corners_top) == 4:
            if len(corners_bottom) == 3:
                # print('side-oblique view (TLR)')
                count_pred = count_points_tlr(keypoints)

            elif len(corners_bottom) == 2:
                # print('side-top view (TL)')
                count_pred = count_points_tl(keypoints)

            elif len(corners_bottom) == 0:
                # print('top view (T)')
                count_pred = count_points_t(keypoints)
            else:
                # fallback: only use top view
                count_pred = count_points_t(keypoints)

        elif len(corners_top) == 3:
            if len(corners_bottom) == 3:
                # print('side-oblique view (LR)')
                count_pred = count_points_lr(keypoints)

        elif len(corners_top) == 2:
            if len(corners_bottom) == 2:
                # print('side view (L)')
                count_pred = count_points_l(keypoints)
            else:
                # fallback: side view again
                count_pred = count_points_l(keypoints)
        else:
            if not quiet:
                print('Cannot estimate view from prediction', file=sys.stderr)
    except Exception as exc:
        if not quiet:
            print(type(exc), exc, file=sys.stderr)

    # print(f'pred = {count_pred}, gt = {count_gt}')

    return count_pred


# if __name__ == '__main__':
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

    # file = next(iter(labels.image.unique()[::-1]))  # debugging
    for file in labels.image.unique():

        print(file)
        file_labels = labels[labels.image == file]
        img_path = input_folder + os.sep + file
        img, a = load_image(img_path, scale, center_crop_fraction)
        img, b = crop_to_prediction(img, (img.shape[0] - model_crop_delta, img.shape[1] - model_crop_delta))

        count_gt = np.array(counts_gt[counts_gt.image == file].cnt)[0]

        count_pred = count_crates(file_labels)

        if count_pred != count_gt:
            print(f'pred = {count_pred}, gt = {count_gt}')

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
            corners_top = get_cat_points(file_labels, 'corner-top', cv=True)
            corners_bottom = get_cat_points(file_labels, 'corner-bottom', cv=True)

            # reorder to descending Y => from lowest to highest (image-coords start in top-left)
            corners_bottom = corners_bottom[np.argsort(corners_bottom[:, 0])[::-1]]

            if len(corners_bottom) == 3:
                # consider any two of the bottom points to be 'front'
                pts_bottom_front = corners_bottom[:2]
            else:
                pts_bottom_front = corners_bottom

            if False:
                dists_bottom_top = cdist(corners_top, pts_bottom_front)
                closest_top_front_indices = np.argmin(dists_bottom_top, axis=0)
                assert len(np.unique(closest_top_front_indices)) == len(closest_top_front_indices), \
                    'bottom-top closest pairs not disjoint'
                # ordering for bottom-to-top-front
                pts_top_front = corners_top[closest_top_front_indices]
                # ^ see doc for shortcomings
            else:
                closest_top_front_indices = closest_pairs_in_order(pts_bottom_front, corners_top, indices_only=True)
                pts_top_front = corners_top[closest_top_front_indices]

            # top non-front points
            pts_top_back = inverse_indexing(corners_top, closest_top_front_indices)  # compare

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

            # x = []  # left-to-right
            # y = []  # front-to-back
            # z = []  # top-to-bottom

            y = axis_direction(pts_top_front, pts_top_back)
            z = axis_direction(pts_bottom_front, pts_top_front)
            x = axis_direction(pts_left, pts_right)

            # print(f'Axes in cv order[y, x]\n{x=}\n{y=}\n{z=}\n')

            pts_left_top = np.vstack([pts_top_front[0], pts_top_back[0]])
            pts_right_top = np.vstack([pts_top_front[1], pts_top_back[1]])

            pts_left_front = np.vstack([pts_bottom_front[0], pts_top_front[0]])
            pts_right_front = np.vstack([pts_bottom_front[1], pts_top_front[1]])

            lines_x = [
                pts_top_front,
                pts_top_back,
                pts_bottom_front
            ]

            lines_y = [
                pts_left_top,
                pts_right_top
            ]

            lines_z = [
                pts_left_front,
                pts_right_front
            ]

            count_pred = final_count(file_labels, lines_x, lines_y, lines_z, show=False, cv=True)

            print('pred =', count_pred)
            count_gt = np.array(counts_gt[counts_gt.image == file].cnt)[0]
            if count_gt != count_pred:
                print('wrong, gt =', count_gt)

            print('...')
            tmp = input()
            if tmp == 'q':
                raise ValueError()


if __name__ == '__main__':
    main()
