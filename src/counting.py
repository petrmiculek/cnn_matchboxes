# stdlib
import os
import sys

# external
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import line as draw_line
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, cosine

# local
from eval_images import crop_to_prediction, load_image
from util import inverse_indexing
from labels import load_labels, resize_labels


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
        ret = np.zeros((pts_from.shape[0]), dtype=np.int)
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

    dists = np.abs(np.cross(p2 - p1, p1 - wrapper))
    if line_len != 0:
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

        # distance factor
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

                err1 = cosine(direction, line1)
                err2 = cosine(direction, line2)
                err_inter = cosine(line1, line2)
                curr_score = err1 + err2 + err_inter

                if curr_score < best_score:
                    best_score = curr_score
                    best_indices = [i, j, m, n]

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


def count_points_tlr(file_labels):
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
    """

    :param corners_bottom:
    :param corners_top_front:
    :return:
    """
    if len(corners_bottom) != len(corners_top_front):
        raise UserWarning(f'Top-bottom matching failed - different number of points '
                          f'{len(corners_bottom)} x {len(corners_top_front)}')
    # get axis shift vector
    up_shift = corners_top_front.mean(axis=0) - corners_bottom.mean(axis=0)

    # shift bottom to top
    pts_bottom_shifted = corners_bottom + up_shift

    # connect shifted bottom to closest top
    top_indices_matched_ordered = closest_pairs_greedy(pts_bottom_shifted, corners_top_front, indices_only=True)

    # check unique indices
    idx_len = top_indices_matched_ordered.shape[0]
    expected_sum = (idx_len * (idx_len - 1) / 2)
    if top_indices_matched_ordered.sum() != expected_sum:
        raise UserWarning(f'Top-bottom matching failed - non-unique pairs: {top_indices_matched_ordered}')

    return top_indices_matched_ordered, up_shift


def count_crates(keypoints, quiet=True):
    corners_top = get_gt_points(keypoints, 'corner-top', cv=False)
    corners_bottom = get_gt_points(keypoints, 'corner-bottom', cv=False)
    count_pred = -1
    try:
        if len(corners_top) >= 4:
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


def main():
    input_folder = 'sirky'  # + '_val'
    labels_path = os.path.join(input_folder, 'labels.csv')
    labels = load_labels(labels_path, use_full_path=False, keep_bg=False)

    scale = 0.25
    center_crop_fraction = 1.0
    model_crop_delta = 63
    labels = resize_labels(labels, scale, model_crop_delta=model_crop_delta, center_crop_fraction=center_crop_fraction)

    counts_gt = pd.read_csv(os.path.join('sirky', 'count.txt'),
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


if __name__ == '__main__':
    """ Run counting on ground-truth keypoints """
    main()
