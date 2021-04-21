# stdlib
import os
import sys
from math import floor, ceil, log10

# external
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from skimage.measure import label as skimage_label

# local
import config
from general import lru_cache
from labels import load_labels, resize_labels
from logs import log_mean_square_error_csv
from show_results import display_predictions
from src_util.general import timing


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
    :param maxes_only:
    :param undecided_only:
    :return:
    """
    import tensorflow as tf
    # ^ since `import tensorflow` starts libcudart, it needs not to be at module-level
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


def mean_square_error_pixelwise(predictions, img_path, file_labels):
    """Calculate MSE metric for full-image prediction

    Metric is calculated per-pixel.
    Consequences:
        Different scale factor is accounted for -> sum / scale**2
        Different model crop rate is not accounted for:
        32x -> -31,
        128x -> -127

        Also, distance from keypoint will be lower at smaller scales.

        Center crop fraction also makes it easier.

    todo calculate MSE from prediction probability instead of argmax

    for all prediction-pixels:
        non-background: distance to closest keypoint of given class ^ 2
                        if no GT keypoint of given category present, fixed penalty

        background: 0.0, if distance to nearest keypoint > min_dist_background
                    else fixed penalty

    """
    min_dist_background = 2.0
    min_dist_background_penalty = 1e3
    no_such_cat_penalty = 1e6  # predicted category not present in GT

    # make prediction one-hot over channels
    pred_argmax = np.argmax(predictions, axis=2)

    # grid for calculating distances
    grid = grid_like(predictions.shape[1], predictions.shape[0])

    mse_categories = {}

    """ background """
    if False:
        file_labels_merged = file_labels[['x', 'y']].to_numpy()

        bg_list = grid[pred_argmax == 0]  # list of 2D coords
        distances = cdist(bg_list, file_labels_merged)

        min_pred_dist = np.min(distances, axis=1)  # distance from predicted bg pixel to nearest GT keypoint
        penalized = min_pred_dist[min_pred_dist < min_dist_background]
        mse_categories['background'] = min_dist_background_penalty * len(penalized) / config.scale ** 2
    else:
        # background ignored for its negligible value and high computation cost
        mse_categories['background'] = 0.0

    """ keypoint categories """
    for i, cat in enumerate(config.class_names[1:], start=1):  # skip 0 == background
        # filter only predictions of given category
        cat_list = grid[pred_argmax == i]

        if cat not in file_labels.category.values:
            mse_categories[cat] = len(cat_list) * no_such_cat_penalty / config.scale ** 2
            continue

        # distance of each predicted point to each GT label
        cat_labels = file_labels[file_labels.category == cat]
        cat_labels = cat_labels[['x', 'y']].to_numpy()
        distances = np.square(cdist(cat_list, cat_labels))
        square_error = np.min(distances, axis=1)

        mse_categories[cat] = np.sum(square_error) / config.scale ** 2

    mse_categories_list = list(mse_categories.values())
    mse_sum = np.sum(mse_categories_list)

    # log to csv
    log_mean_square_error_csv(config.model_name, img_path, mse_sum, mse_categories_list)

    return mse_sum, mse_categories


def full_prediction(base_model, file_labels, img_path, output_location=None,
                    show=True, maxes_only=False, undecided_only=False):
    """Show predicted heatmaps for full image

    :param base_model:
    :param file_labels:
    :param img_path:
    :param output_location:
    :param show:
    :param maxes_only: Show only the top predicted category per point.
    :param undecided_only: Show only predictions that were
    """

    img, orig_size = load_image(img_path, config.scale, config.center_crop_fraction)

    predictions = make_prediction(base_model, img, maxes_only, undecided_only)

    # todo
    # temporary - save predictions
    # os.makedirs('preds', exist_ok=True)
    # preds_file_path = os.path.join('preds', img_path.split('/')[-1] + '.npy')
    #
    # with open(preds_file_path, 'wb') as f:
    #     noinspection PyTypeChecker
    #     np.save(f, predictions)

    # Model prediction is cropped, adjust image accordingly
    img, model_crop_delta = crop_to_prediction(img, predictions.shape)

    category_titles = config.class_names

    pix_mse = -1
    point_mse = -1
    count_mae = np.nan
    if undecided_only:
        plots_title = 'undecided'
    elif maxes_only:
        plots_title = 'maxes'
    else:
        # try:
        if True:
            # pixel-wise
            pix_mse, mse_pix_categories = mean_square_error_pixelwise(predictions, img_path, file_labels)

            # category_titles = ['{}: 1e{:0.2g}'.format(cat, log10(cat_mse + 1)) for cat, cat_mse in
            #                    mse_pix_categories.items()]
            # plots_title = '{:0.2g}M'.format(mse_pix_total / 1e6)

            # point-wise
            point_mse, mse_categories, count_mae = mean_square_error_pointwise(predictions, file_labels, show=show)

            category_titles = ['{}: {:.1e}'.format(cat, cat_mse) for cat, cat_mse in
                               mse_categories.items()]
            plots_title = 'dist_mse: {:.1e}, count_mae: {:0.2g}'.format(point_mse, count_mae)

            # show_mse_pixelwise_location(predictions, img, img_path, file_labels, output_location=output_location,
            #                             show=False)
        # except Exception as ex:
        #     plots_title = ''
        #     print(ex, file=sys.stderr)

    assert len(category_titles) >= 8, '{}, {}, {}'.format(len(category_titles), point_mse, pix_mse)
    # save_predictions_cv(predictions, img, img_path, output_location)
    display_predictions(predictions, img, img_path, category_titles, plots_title,
                        show=show, output_location=output_location, superimpose=True)

    return pix_mse, point_mse, count_mae


def full_prediction_all(base_model, val=True,
                        undecided_only=False, maxes_only=False,
                        output_location=None, show=True):
    images_dir = 'sirky' + '_val' * val
    labels = load_labels(images_dir + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)
    labels = resize_labels(labels, config.scale, config.train_dim - 1, config.center_crop_fraction)

    if output_location:
        output_location = os.path.join(output_location, 'heatmaps')
        os.makedirs(output_location, exist_ok=True)

    point_mse_list = []
    pix_mse_list = []
    count_mae_list = []
    for file in pd.unique(labels['image']):
        file_labels = labels[labels.image == file]
        pix_mse, point_mse, count_mae = \
            full_prediction(base_model, file_labels, img_path=images_dir + os.sep + file,
                            output_location=output_location, show=show,
                            undecided_only=undecided_only, maxes_only=maxes_only)

        pix_mse_list.append(pix_mse)
        point_mse_list.append(point_mse)
        count_mae_list.append(count_mae)

    pix_mse = np.mean(pix_mse_list)
    point_mse = np.mean(point_mse_list)
    count_mae = np.mean(count_mae_list) if len(count_mae_list) > 0 else 0
    return pix_mse, point_mse, count_mae


def mean_square_error_pointwise(predictions, file_labels, show=False):
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

    # show = False


    prediction_threshold = 0.9
    min_blob_size = 160 * config.scale ** 2

    img_file_name = file_labels.iloc[0].image

    # distance mse mean calculation
    mse_cat_list = []

    # per-category distance mse for plotting
    mse_cat_dict = {'background': 0.0}

    # keypoint counting mae
    count_error_categories = []

    # for development purposes only
    blob_sizes = []

    if show:
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # thresholding -> 0, 1
    predictions[:, :, 1:] = (predictions[:, :, 1:] >= prediction_threshold).astype(np.float64)

    prediction_argmax = np.argmax(predictions, axis=2)

    # find blobs in any non-background prediction pixels (ignoring keypoint category)
    blobs, num_blobs = skimage_label(prediction_argmax > 0, return_num=True, connectivity=2)

    pts_pred = []
    pts_pred_categories = []

    # create a keypoint from blob pixels
    for blob in range(1, num_blobs + 1):
        blob_indices = np.argwhere(blobs == blob)
        if len(blob_indices) < min_blob_size:
            continue

        cats, support = np.unique(prediction_argmax[blobs == blob], return_counts=True)
        blob_sizes.append(len(blob_indices))

        center = np.mean(blob_indices, axis=0).astype(np.int)
        winning_category = cats[np.argmax(support)]
        pts_pred.append(center)
        pts_pred_categories.append(winning_category)

    if len(pts_pred) > 0:
        pts_pred = np.flip(pts_pred, axis=1)  # yx -> xy
    else:
        # fix: empty list does not have dimensions like (n, 2)
        pts_pred = np.array((0, 2))

    pts_pred_categories = np.array(pts_pred_categories)

    for cat in range(predictions.shape[2]):
        if show:
            ax = axes[cat // 3, cat % 3]
            ax.set_title(config.class_names[cat])
            ax.imshow(predictions[:, :, cat])

        if cat == 0:
            continue

        pts_gt_cat = file_labels[file_labels.category == config.class_names[cat]][['x', 'y']].to_numpy()
        pts_pred_cat = pts_pred[pts_pred_categories == cat]

        # if len(pts_pred_cat) != len(pts_gt_cat):
        #     print('\t', config.class_names[cat], len(pts_pred_cat), '->', len(pts_gt_cat))
        mse = mse_value(pts_gt_cat, pts_pred_cat)

        if show:
            if len(pts_gt_cat) > 0:
                ax.scatter(pts_gt_cat[:, 0], pts_gt_cat[:, 1], marker='o')

            if len(pts_pred_cat) > 0:
                ax.scatter(pts_pred_cat[:, 0], pts_pred_cat[:, 1], marker='x')

            ax.set_title('{}: {:.2e}'.format(config.class_names[cat], mse))

        mse_cat_dict[config.class_names[cat]] = mse
        mse_cat_list.append(mse)
        count_error_categories.append(pts_pred_cat.shape[0] - pts_gt_cat.shape[0])

    mse_total = np.mean(mse_cat_list)
    count_mae = np.mean(np.abs(count_error_categories))

    if show:
        fig.legend(['ground-truth', 'prediction'])
        if len(pts_pred) > 0:
            axes[2, 2].scatter(pts_pred[:, 0], pts_pred[:, 1], marker='*', color='r')

        axes[2, 2].set_xlim(0, predictions.shape[1])
        axes[2, 2].set_ylim(0, predictions.shape[0])
        axes[2, 2].invert_yaxis()

        # original image
        img_path = os.path.join('sirky', img_file_name)
        if os.path.isfile(img_path):
            img, _ = load_image(img_path, config.scale, config.center_crop_fraction)
            img, _ = crop_to_prediction(img, predictions.shape)
            axes[2, 2].imshow(img)
        else:
            # print('not found:', img_path)
            pass

        fig.suptitle('{}: {:.2e}'.format(img_file_name, mse_total))
        fig.tight_layout()
        fig.show()

    # print('blob size:', np.mean(blob_sizes).astype(np.int))

    return mse_total, mse_cat_dict, count_mae


def show_mse_pixelwise_location(predictions, img, img_path, file_labels, output_location=None, show=True):
    """Show contributions to MSE in full image predictions

    :param predictions: (W, H, C) prediction tensor
    :param img:
    :param img_path:
    :param file_labels:
    :param show:
    :param output_location:

    inspiration for grid operations (4-th answer):
    https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy

    idea: grid distance to keypoint could be cached and loaded from a pickle

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

        # normalizing ~~
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
