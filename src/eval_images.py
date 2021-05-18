# stdlib
import os
import sys
from math import floor, ceil

# external
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from skimage.measure import label as skimage_label

# local
import config
from util import inverse_indexing, lru_cache
from labels import get_gt_count, load_labels, resize_labels
from logs import log_mean_square_error_csv
from display import display_predictions, display_keypoints


def load_image(img_path, scale=1.0, center_crop_fraction=1.0):
    """Open and Process image

    OpenCV coordinates [y, x] for image
    NumPy coordinates for orig_size [x, y, c=channels]

    :param img_path: Path for loading image
    :param scale: Rescale image factor
    :param center_crop_fraction: Crop out rectangle with sides equal to a given fraction of original image
    :return: image [y, x], original image shape [x, y, c]
    """
    assert os.path.isfile(img_path), \
        'image path "{}" invalid'.format(img_path)

    img = cv.imread(img_path)

    orig_size = np.array((img.shape[1], img.shape[0], img.shape[2]))  # reversed indices

    if center_crop_fraction != 1.0:
        # crop from center
        low = (1 - center_crop_fraction) / 2
        high = (1 + center_crop_fraction) / 2

        img = img[int(img.shape[0] * low): int(img.shape[0] * high),
                  int(img.shape[1] * low): int(img.shape[1] * high),
                  :]

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                    interpolation=cv.INTER_AREA)  # reversed indices, OK
    return img, orig_size


def make_prediction(base_model, img, maxes_only=False, undecided_only=False):
    """Make full-image prediction

    Not worth trying to turn this into tf.ops, since it requires further workarounds for assigning to tensors

    :param base_model:
    :param img: img as batch
    :param maxes_only: use only predictions argmax
    :param undecided_only: use only predictions with low confidence values
    :return:
    """
    import tensorflow as tf
    # ^ since `import tensorflow` starts libcudart, it is better not to put the import at module-level
    # it makes a difference when a non-tf function from this module is imported in a non-tf environment

    predictions_raw = base_model.predict(tf.convert_to_tensor(tf.expand_dims(img, 0), dtype=tf.uint8))
    predictions = tf.squeeze(predictions_raw).numpy()

    if maxes_only and not undecided_only:
        maxes = np.zeros(predictions.shape)
        max_indexes = np.argmax(predictions, axis=-1)
        maxes[np.arange(predictions.shape[0])[:, None],  # in every row
              np.arange(predictions.shape[1]),  # in every column
              max_indexes] = 1  # the maximum-predicted category (=channel) is set to 1

        predictions = maxes

    if undecided_only and not maxes_only:
        decisive_values = np.max(predictions, axis=2)  # per-pixel
        decisive_indices = np.where(decisive_values < 0.5, True, False)
        decisive_indices = np.stack((decisive_indices,) * 8, axis=-1)
        undecided = np.where(decisive_indices, predictions, 0)

        predictions = undecided

    return predictions


def crop_to_prediction(img, pred_shape):
    """

    returns a view, could mess up original image by being mutated later on

    :param img: image to be cropped
    :param pred_shape: 2d or 3d
    :return:
    """
    assert img.shape[0] - pred_shape[0] == img.shape[1] - pred_shape[1]
    model_crop_delta = img.shape[0] - pred_shape[0]
    low = floor(model_crop_delta / 2)
    high = ceil(model_crop_delta / 2)

    return img[low:-high, low:-high], model_crop_delta


@lru_cache
def grid_like(x, y):
    grid = np.meshgrid(np.arange(x), np.arange(y))
    grid = np.vstack([grid])
    grid = np.transpose(grid, (1, 2, 0))
    return grid


def mse_pixelwise(predictions, img_path, file_labels):
    """Calculate pixel-wise MSE metric for full-image prediction

    Metric is calculated per-pixel.
    Consequences:
        Different scale factor is accounted for -> sum / scale**2
        Different model crop rate is not accounted for:
        32x -> -31,
        128x -> -127

        Also, distance from keypoint will be lower at smaller scales.

        Center crop fraction also makes it easier.


    for all prediction-pixels:
        non-background: square distance to the closest keypoint of given class
                        if no GT keypoint of given category present, fixed penalty

        background: 0.0, if distance to nearest keypoint > min_dist_background
                    else fixed penalty

    :return: MSE sum over categories (mean weighted by category support),
             MSE per category (weighted by category support)

    """
    min_dist_background = 2.0
    min_dist_background_penalty = 1e3
    no_such_cat_penalty = 1e3  # predicted category not present in GT

    # make prediction one-hot over channels
    pred_argmax = np.argmax(predictions, axis=2)

    # grid for calculating distances
    grid = grid_like(predictions.shape[1], predictions.shape[0])

    mse_categories = []
    support = []

    """ background """
    if False:
        # unused
        file_labels_merged = file_labels[['x', 'y']].to_numpy()

        bg_list = grid[pred_argmax == 0]  # list of 2D coords
        distances = cdist(bg_list, file_labels_merged)

        min_pred_dist = np.min(distances, axis=1)  # distance from predicted bg pixel to nearest GT keypoint
        penalized = min_pred_dist[min_pred_dist < min_dist_background]
        mse_categories.append(min_dist_background_penalty * len(penalized))
    else:
        # background ignored for its negligible value and high computation cost
        mse_categories.append(0.0)
        support.append(0)

    """ keypoint categories """
    for i, cat in enumerate(config.class_names[1:], start=1):  # skip 0 == background
        # filter only predictions of given category
        cat_list = grid[pred_argmax == i]

        support.append(len(cat_list))
        if cat not in file_labels.category.values:
            mse_categories.append(len(cat_list) * no_such_cat_penalty)
            continue

        # distance of each predicted point to each GT label
        cat_labels = file_labels[file_labels.category == cat]
        cat_labels = cat_labels[['x', 'y']].to_numpy()
        distances = np.square(cdist(cat_list, cat_labels))
        square_error = np.min(distances, axis=1)
        if len(square_error) == 0:
            # false negative, cannot evaluate per pixel, ignored
            square_error = 0

        mse_categories.append(np.mean(square_error))

    mse_categories = np.array(mse_categories)
    support = np.array(support)

    mse_categories /= config.scale ** 2  # normalizing factor

    if len(support) > 0:
        mse_categories = mse_categories * support / np.sum(support)
    mse_sum = np.sum(mse_categories)

    # log to csv
    log_mean_square_error_csv(config.model_name, img_path, mse_sum, mse_categories)

    """
    output:
    
    total: sum(total_square_error * support)
    per_category: total_square_error * support / sum(support)
    
    """
    mse_dict = dict(zip(config.class_names, mse_categories))
    return mse_sum, mse_dict


def mse_pointwise(predictions, img, keypoints, kp_categories, file_labels, show=False):
    """Keypoint-wise Mean Square Error + plotting keypoints

    :param predictions:
    :param img:
    :param keypoints:
    :param kp_categories:
    :param file_labels:
    :param show:
    :return:
    """

    def mse_value(pts_gt, pts_pred):
        false_positive_penalty = 1e4
        false_negative_penalty = 1e4

        if len(pts_gt) == len(pts_pred) == 0:
            mse_val = 0.0
        elif len(pts_gt) == 0 and len(pts_pred) > 0:
            # no ground-truth
            mse_val = len(pts_pred) * false_positive_penalty
        elif len(pts_gt) > 0 and len(pts_pred) == 0:
            # no prediction
            mse_val = len(pts_gt) * false_negative_penalty
        else:
            dists = cdist(pts_gt, pts_pred)
            preds_errors = np.min(dists, axis=0)
            mse_val = np.mean(np.square(preds_errors)) / config.scale

            # false-negative penalty
            mse_val += np.max([0, len(pts_gt) - len(pts_pred)]) * false_negative_penalty

        return mse_val

    img_file_name = file_labels.iloc[0].image

    # distance mse mean calculation
    mse_cat_list = []

    # per-category distance mse for plotting
    dist_mse_cat_dict = {'background': 0.0}

    # keypoint counting mae
    count_error_categories = []

    if show:
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for i, cat in enumerate(config.class_names):
        if show:
            ax = axes[i // 3, i % 3]
            ax.set_title(cat)
            ax.imshow(predictions[:, :, i])
            ax.axis('off')

        if i == 0:
            continue

        pts_gt_cat = file_labels[file_labels.category == cat][['x', 'y']].to_numpy()
        pts_pred_cat = keypoints[kp_categories == i]

        mse = mse_value(pts_gt_cat, pts_pred_cat)

        if show:
            if len(pts_gt_cat) > 0:
                ax.scatter(pts_gt_cat[:, 0], pts_gt_cat[:, 1], color='orange', marker='o')

            if len(pts_pred_cat) > 0:
                ax.scatter(pts_pred_cat[:, 0], pts_pred_cat[:, 1], color='green', marker='+')

            ax.set_title('{}\n{:.2e}'.format(cat, mse))

        dist_mse_cat_dict[cat] = mse
        mse_cat_list.append(mse)
        count_error_categories.append(pts_pred_cat.shape[0] - pts_gt_cat.shape[0])

    dist_mse_total = np.mean(mse_cat_list)
    keypoint_count_mae = np.sum(np.abs(count_error_categories))

    if show:
        fig.legend(['ground-truth', 'prediction'])
        if len(keypoints) > 0 and np.array(keypoints).ndim == 2:
            axes[2, 2].scatter(keypoints[:, 0], keypoints[:, 1], marker='+', color='lime')

        axes[2, 2].set_xlim(0, img.shape[1])
        axes[2, 2].set_ylim(0, img.shape[0])
        axes[2, 2].invert_yaxis()
        axes[2, 2].imshow(img)

        fig.suptitle('Keypoints {}\npoint_mae: {:.2e}'.format(img_file_name, dist_mse_total))
        fig.tight_layout()
        fig.show()

    return dist_mse_total, dist_mse_cat_dict, keypoint_count_mae


def remove_keypoint_outliers(points, categories, threshold_dist=1.8):
    dists = cdist(points, points)

    threshold_dist *= np.mean(dists)

    far_from_all = np.argwhere(np.mean(dists, axis=0) > threshold_dist)

    points = inverse_indexing(points, far_from_all)
    categories = inverse_indexing(categories, far_from_all)

    return points, categories


def prediction_to_keypoints(predictions, min_blob_size=None, prediction_threshold=0.9):
    """Convert prediction to keypoints

    Threshold prediction
    Find connected components
    Merge neighboring components

    idea: split blob if several parts are > min_blob_size

    :return: keypoints, categories (both as np.arrays)
    """
    if min_blob_size is None:
        min_blob_size = 160 * config.scale ** 2

    # thresholding -> 0, 1
    predictions[:, :, 1:] = (predictions[:, :, 1:] >= prediction_threshold).astype(np.float64)
    prediction_argmax = np.argmax(predictions, axis=2)

    # find blobs in non-background prediction pixels (any keypoint category)
    blobs, num_blobs = skimage_label(prediction_argmax > 0, return_num=True, connectivity=2)
    points = []
    points_categories = []

    # create a keypoint from blob pixels
    blob_sizes = []

    for blob in range(1, num_blobs + 1):
        blob_indices = np.argwhere(blobs == blob)
        if len(blob_indices) < min_blob_size:
            continue

        cats, support = np.unique(prediction_argmax[blobs == blob], return_counts=True)
        blob_sizes.append(len(blob_indices))

        center = np.mean(blob_indices, axis=0).astype(np.int)
        winning_category = cats[np.argmax(support)]
        points.append(center)
        points_categories.append(winning_category)

    if len(points) > 0:
        points = np.flip(points, axis=1)  # yx -> xy
    else:
        # fix: empty list does not have dimensions like (n, 2)
        points = np.array((0, 2))

    points_categories = np.array(points_categories)
    # print('blob size:', np.mean(blob_sizes).astype(np.int))

    return points, points_categories  # , blob_sizes


def eval_full_prediction(base_model, file_labels, img_path, output_location=None, show=True):
    """Show predicted heatmaps for full image and evaluate prediction

    :param base_model:
    :param file_labels:
    :param img_path:
    :param output_location:
    :param show:
    """
    from counting import count_crates  # avoids circular import dependencies

    img, orig_size = load_image(img_path, config.scale, config.center_crop_fraction)

    if min(img.shape[0], img.shape[1]) < config.train_dim:
        err = f'Image too small for prediction [{img.shape[0]},{img.shape[1]}]\n' + \
              f'must be at least as big as training dimension = {config.train_dim}\n' + \
              f'image_path:{img_path}\n'
        raise UserWarning(err)

    predictions = make_prediction(base_model, img)

    # Model prediction is cropped, adjust image accordingly
    img, model_crop_delta = crop_to_prediction(img, predictions.shape)

    category_titles = config.class_names

    pix_mse = -1
    dist_mse = -1
    keypoint_count_mae = np.nan
    crate_count_error = np.nan
    plots_title = ''

    try:
        crate_count_gt = get_gt_count(img_path)

        # pixel-wise
        pix_mse, pix_mse_categories = mse_pixelwise(predictions, img_path, file_labels)

        keypoints_pred, kp_pred_categories = prediction_to_keypoints(predictions)
        keypoints_pred, kp_pred_categories = remove_keypoint_outliers(keypoints_pred, kp_pred_categories)

        # keypoint-wise
        dist_mse, dist_mse_categories, keypoint_count_mae = \
            mse_pointwise(predictions, img, keypoints_pred, kp_pred_categories, file_labels, show=show)

        # crate-counting
        df = pd.DataFrame(np.c_[keypoints_pred, kp_pred_categories], columns=['x', 'y', 'category'])
        label_dict = dict(enumerate(config.class_names))
        df['category'] = df['category'].map(label_dict)
        crate_count_pred = count_crates(df, quiet=True)
        if crate_count_pred != -1:
            crate_count_error = np.abs(crate_count_pred - crate_count_gt)

        category_titles = ['{}: {:.1e}'.format(cat, cat_mse) for cat, cat_mse in dist_mse_categories.items()]
        plots_title = f'pix_mse: {pix_mse:.1e}, dist_mse: {dist_mse:.1e}, count_mae: {keypoint_count_mae:0.2g}, ' \
                      f'Pred: {crate_count_pred:0.2g}, GT: {crate_count_gt:0.2g}'

        kp_plots_title = f'Pred: {crate_count_pred:0.2g}, GT: {crate_count_gt:0.2g}'

        display_keypoints((keypoints_pred, kp_pred_categories), img, img_path, config.class_names, title=kp_plots_title,
                          show=show, output_location=output_location)

        # unused
        # show_mse_pixelwise_location(predictions, img, img_path, file_labels, output_location, show=False)
    except Exception as ex:
        print(ex, file=sys.stderr)

    assert len(category_titles) >= 8, '{}, {}, {}'.format(len(category_titles), dist_mse, pix_mse)
    # save_predictions_cv(predictions, img, img_path, output_location)
    display_predictions(predictions, img, img_path, category_titles, plots_title,
                        show=show, output_location=output_location, superimpose=True)


    return pix_mse, dist_mse, keypoint_count_mae, crate_count_error


def eval_full_predictions_all(base_model, val=True, output_location=None, show=True):
    """Evaluate prediction on all images in the training/validation dataset

    :param base_model:
    :param val:
    :param output_location:
    :param show:
    :return:
    """

    images_dir = 'sirky' + '_val' * val
    labels = load_labels(images_dir + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)
    labels = resize_labels(labels, config.scale, config.train_dim - 1, config.center_crop_fraction)

    if output_location:
        output_location = os.path.join(output_location, 'heatmaps')
        os.makedirs(output_location, exist_ok=True)

    pix_mse_list = []
    dist_mse_list = []
    count_mae_list = []
    crate_count_err_list = []
    for file in pd.unique(labels['image']):
        file_labels = labels[labels.image == file]
        pix_mse, dist_mse, keypoint_count_mae, crate_count_err = \
            eval_full_prediction(base_model, file_labels, img_path=images_dir + os.sep + file,
                                 output_location=output_location, show=show)

        pix_mse_list.append(pix_mse)
        dist_mse_list.append(dist_mse)
        count_mae_list.append(keypoint_count_mae)
        crate_count_err_list.append(crate_count_err)

    pix_mse = np.nanmean(pix_mse_list)
    dist_mse = np.nanmean(dist_mse_list)
    keypoint_count_mae = np.nanmean(count_mae_list) if len(count_mae_list) > 0 else 0
    crate_count_failrate = np.sum(np.isnan(crate_count_err_list)) / len(crate_count_err_list)
    crate_count_mae = np.nanmean(crate_count_err_list) if crate_count_failrate < 1 else np.nan

    return pix_mse, dist_mse, keypoint_count_mae, crate_count_mae, crate_count_failrate


def show_mse_pixelwise_location(predictions, img, img_path, file_labels, output_location=None, show=True):
    """Show contributions to MSE in full image predictions

    :param predictions: (W, H, C) prediction tensor
    :param img:
    :param img_path:
    :param file_labels:
    :param show:
    :param output_location:
    """
    if not show and output_location is None:
        return

    def dst_pt2grid(pt):
        return np.square(np.hypot(gx - pt[0], gy - pt[1]))

    gx, gy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    flat_penalty = np.zeros((img.shape[0], img.shape[1])) + 100

    cat_distances = []
    for cat in config.class_names:
        if cat not in file_labels.category.values:
            if cat == 'background':
                cat_distances.append(np.zeros((img.shape[0], img.shape[1])))
            else:
                cat_distances.append(flat_penalty)
            continue

        cat_labels = file_labels[file_labels.category == cat]
        cat_labels = np.array([cat_labels.x, cat_labels.y]).T

        distance_layers = list(map(dst_pt2grid, cat_labels))
        distance_layers = np.vstack([distance_layers])

        # normalizing
        distance_layers = distance_layers / np.sqrt(np.max(distance_layers))

        min_distance = np.min(distance_layers, axis=0)  # minimum distance to any keypoint
        cat_distances.append(min_distance)

    cat_distances = np.dstack(cat_distances)
    titles = np.append(config.class_names, 'full_image')

    # elementwise multiply predictions with cat_distances
    tst = predictions * cat_distances

    # tst expected to be in [0 .. 1] range,
    tst /= np.max(tst)
    # but it would be barely visible like that
    tst *= 10

    if output_location:
        output_location = os.path.join(output_location, 'err_distribution')

    # show/save with matplotlib
    display_predictions(tst, img, img_path, titles, output_location=output_location, show=show)

    # save with OpenCV - unused
    # save_predictions_cv(tst, img, img_path, output_location)
