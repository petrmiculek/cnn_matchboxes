# stdlib
import os

# external
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage.measure import label as skimage_label
# from skimage.io import imread
# from skimage.transform import resize
import seaborn as sns

# local
# from src.display import display_predictions
import config
from src_util.labels import load_labels, resize_labels
from src.eval_images import load_image, crop_to_prediction


def mse_value(pts_gt, pts_pred):
    false_positive_penalty = 1e4
    false_negative_penalty = 1e4

    if len(pts_gt) == len(pts_pred) == 0:
        mse = 0.0
    elif len(pts_gt) == 0 and len(pts_pred) > 0:
        # no ground-truth
        mse = len(pts_pred) * false_positive_penalty
    elif len(pts_gt) > 0 and len(pts_pred) == 0:
        # no prediction
        mse = len(pts_gt) * false_negative_penalty
    else:
        dists = cdist(pts_gt, pts_pred)
        preds_errors = np.min(dists, axis=0)
        mse = np.mean(np.square(preds_errors)) / config.scale

    return mse


def mean_square_error(pts_gt, pts_pred):
    """
    Single category only

    For each prediction point, the error is the distance to the nearest ground-truth keypoint.

    Ignoring many-to-one (preds to gt) matching

    this is analogous to the pixel-wise MSE in eval_images

    :param pts_gt:
    :param pts_pred:
    :return:
    """
    from counting import inverse_indexing, closest_pairs_in_order, closest_pairs_greedy

    assert pts_gt.ndim == pts_pred.ndim == 2
    assert pts_gt.shape[1] == pts_gt.shape[1] == 2

    mse = mse_value(pts_gt, pts_pred)

    if False:
        t = 100  # distance threshold

        false_pos_indices = np.argwhere(np.min(dists, axis=0) > t).flatten()
        false_neg_indices = np.argwhere(np.min(dists, axis=1) > t).flatten()

        fp = pts_pred[false_pos_indices]
        fn = pts_gt[false_neg_indices]

        if len(false_neg_indices) + len(false_pos_indices) == len(pts_gt) + len(pts_pred):
            print('all wrong')

        matched_pred = inverse_index(pts_pred, false_pos_indices)
        matched_gt = inverse_index(pts_gt, false_neg_indices)

        # other = np.r_[matched_gt, matched_pred]
        # dists_matched = cdist(matched_gt, matched_pred)
        # neighbor_preds = (dists_matched < t).sum(axis=1)

    if False:
        tp_ = len(matched_gt)
        fp_ = len(fp)
        fn_ = len(fn)
        print('tp     =', tp_)
        print('fp     =', fp_)
        print('fn     =', fn_)

        print('prec   =', tp_ / (tp_ + fp_))
        print('recall =', tp_ / (tp_ + fn_))

        plt.scatter(fp[:, 0], fp[:, 1], marker='x')
        plt.scatter(fn[:, 0], fn[:, 1], marker='+')
        plt.scatter(other[:, 0], other[:, 1], marker='o')
        plt.show()

    return mse_value


if __name__ == '__main__':
    config.center_crop_fraction = 0.5
    config.scale = 0.5
    config.train_dim = 64
    model_crop_delta = config.train_dim - 1

    prediction_threshold = 0.9
    min_blob_size = 160 * config.scale ** 2
    predictions_dir = 'preds'
    images_dir = 'sirky'
    labels = load_labels(os.path.join(images_dir, 'labels.csv'), False, False)
    labels = resize_labels(labels, config.scale, model_crop_delta, config.center_crop_fraction)
    class_names = ['background'] + sorted(pd.unique(labels.category))

    show = True
    output_location = True

    predictions = []
    file_names = []

    blob_sizes = []
    multiblobs = []

    mse_totals = []
    count_error_categories_total = []

    for file in os.listdir(predictions_dir):
        suffix = file[file.rfind('.'):]
        if suffix != '.npy':
            continue

        pred = np.load(os.path.join(predictions_dir, file), allow_pickle=True)
        predictions.append(pred)

        img_file_name = file[:file.rfind('.')]  # remove .npy
        if not os.path.isfile(os.path.join(images_dir, img_file_name)):
            continue

        file_names.append(file)
        file_labels = labels[labels.image == img_file_name]

        print(img_file_name)

        mse_cat_dict = {}
        mse_cat_list = []
        count_error_categories = []
        if show:
            fig, axes = plt.subplots(3, 3, figsize=(12, 10))

        """ Approach 2: Merging neighboring blobs"""
        # thresholding -> 0, 1
        pred[:, :, 1:] = (pred[:, :, 1:] >= prediction_threshold).astype(np.float64)

        prediction_argmax = np.argmax(pred, axis=2)

        blobs, num_blobs = skimage_label(prediction_argmax > 0, return_num=True, connectivity=2)

        pts_pred = []
        pts_pred_categories = []

        for blob in range(1, num_blobs + 1):
            blob_indices = np.argwhere(blobs == blob)
            if len(blob_indices) < min_blob_size:
                continue

            cats, support = np.unique(prediction_argmax[blobs == blob], return_counts=True)
            blob_sizes.append(len(blob_indices))
            if len(support) > 1:
                multiblobs.append((cats, support))
                # print(len(support), np.max(blob_indices, axis=0) - np.min(blob_indices, axis=0))

            center = np.mean(blob_indices, axis=0).astype(np.int)  # check axis
            winning_category = cats[np.argmax(support)]
            pts_pred.append(center)
            pts_pred_categories.append(winning_category)

        if len(pts_pred) > 0:
            pts_pred = np.flip(pts_pred, axis=1)  # yx -> xy
        else:
            # fix: empty list does not have dimensions like (n, 2)
            pts_pred = np.array((0, 2))

        pts_pred_categories = np.array(pts_pred_categories)

        for cat in range(pred.shape[2]):

            if show:
                ax = axes[cat // 3, cat % 3]
                ax.set_title(class_names[cat])
                ax.imshow(pred[:, :, cat])

            if cat == 0:
                continue

            pts_gt_cat = file_labels[file_labels.category == class_names[cat]][['x', 'y']].to_numpy()
            pts_pred_cat = pts_pred[pts_pred_categories == cat]

            if len(pts_pred_cat) != len(pts_gt_cat):
                print('\t', class_names[cat], len(pts_pred_cat), '->', len(pts_gt_cat))

            if show:
                ax.scatter(pts_gt_cat[:, 0], pts_gt_cat[:, 1], marker='o')
                ax.scatter(pts_pred_cat[:, 0], pts_pred_cat[:, 1], marker='x')

            mse = mse_value(pts_gt_cat, pts_pred_cat)
            ax.set_title('{}: {:.2e}'.format(class_names[cat], mse))

            mse_cat_dict[class_names[cat]] = mse
            mse_cat_list.append(mse)
            count_error_categories.append(pts_pred_cat.shape[0] - pts_gt_cat.shape[0])

        # print("{:.2e}".format(mse_value))
        # print(file)
        # for cat, val in mse_categories.items():
        #     print('\t{}: {:.2e}'.format(cat, val))

        count_error_categories_total.append(count_error_categories)
        mse_file_total = np.mean(mse_cat_list)
        mse_totals.append(mse_file_total)

        print(img_file_name, '{:.2e}'.format(mse_file_total))
        # print(pts_pred.shape[0], len(file_labels))

        if show:
            fig.legend(['prediction', 'ground-truth'])  # todo check if matching

            # original image
            img_path = os.path.join(images_dir, img_file_name)
            if os.path.isfile(img_path):
                img, _ = load_image(img_path, config.scale, config.center_crop_fraction)
                img, _ = crop_to_prediction(img, pred.shape)
                axes[2, 2].imshow(img)
                axes[2, 2].scatter(pts_pred[:, 0], pts_pred[:, 1], marker='*')
            else:
                print('not found:', img_path)

            fig.suptitle('{}: {:.2e}'.format(img_file_name, mse_file_total))
            fig.tight_layout()
            fig.show()

        if output_location:
            fig.savefig(os.path.join(predictions_dir, img_file_name + '_plot_mse.png'), bbox_inches='tight')
        # file-loop end

    print('Total mean MSE: {:.2e}'.format(np.mean(mse_totals)))
    print('Total count MAE: {:.2e}'.format(np.mean(count_error_categories_total)))
    print('Per category count MAE')
    print(np.vstack(count_error_categories_total).mean(axis=0))

    sns.histplot(blob_sizes, binwidth=5)
    plt.xlim(0, 400)
    plt.show()
