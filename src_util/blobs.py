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
# from src.show_results import display_predictions
import config
from src_util.labels import load_labels, rescale_labels
from src.eval_images import load_image, crop_to_prediction


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
    from hough import inverse_index, closest_pairs_in_order, closest_pairs_greedy

    assert pts_gt.ndim == pts_pred.ndim == 2
    assert pts_gt.shape[1] == pts_gt.shape[1] == 2

    false_positive_penalty = 1e4
    false_negative_penalty = 1e4

    if len(pts_gt) == len(pts_pred) == 0:
        return 0.0

    elif len(pts_gt) == 0 and len(pts_pred) > 0:
        return len(pts_pred) * false_positive_penalty

    elif len(pts_gt) > 0 and len(pts_pred) == 0:
        return len(pts_gt) * false_negative_penalty

    dists = cdist(pts_gt, pts_pred)

    preds_errors = np.min(dists, axis=0)
    mse_value = np.mean(np.square(preds_errors))

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
    labels = rescale_labels(labels, config.scale, model_crop_delta, config.center_crop_fraction)
    class_names = ['background'] + sorted(pd.unique(labels.category))

    show = True
    output_location = False

    predictions = []
    file_names = []

    blob_sizes = []
    multiblobs = []

    mse_totals = []

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

        # if '121303' in img_file_name:
        #     raise ValueError()

        print(img_file_name)

        mse_cat_dict = {}
        mse_cat_list = []
        count_error_categories = []
        if show:
            fig, axes = plt.subplots(3, 3, figsize=(12, 10))

        """ Approach 2: Merging neighboring blobs"""
        if True:
            # thresholding -> 0, 1
            pred[:, :, 1:] = (pred[:, :, 1:] >= prediction_threshold).astype(np.float64)

            prediction_argmax = np.argmax(pred, axis=2)

            blobs, num_blobs = skimage_label(prediction_argmax > 0, return_num=True, connectivity=2)

            pts_pred_center = []
            pts_pred_cat = []

            for blob in range(1, num_blobs + 1):
                blob_indices = np.argwhere(blobs == blob)
                if len(blob_indices) < min_blob_size:
                    continue
                cats, support = np.unique(prediction_argmax[blobs == blob], return_counts=True)
                blob_sizes.append(len(blob_indices))
                if len(support) > 1:
                    multiblobs.append((cats, support))
                    print(len(support), np.max(blob_indices, axis=0) - np.min(blob_indices, axis=0))

                center = np.mean(blob_indices, axis=0).astype(np.int)  # check axis
                winning_category = cats[np.argmax(support)]
                pts_pred_center.append(center)
                pts_pred_cat.append(winning_category)

            pts_pred_center = np.array(pts_pred_center)
            pts_pred_cat = np.array(pts_pred_cat)

            old_blobs_counts = []
            for cat in range(pred.shape[2]):

                if show:
                    ax = axes[cat // 3, cat % 3]
                    ax.set_title(class_names[cat])
                    ax.imshow(pred[:, :, cat])

                if cat == 0:
                    continue

                old_blobs, old_num_blobs = skimage_label(pred[:, :, cat], return_num=True, connectivity=2)
                old_blobs_counts.append(old_num_blobs)

                pts_gt = file_labels[file_labels.category == class_names[cat]][['x', 'y']].to_numpy()
                pts_pred = pts_pred_center[pts_pred_cat == cat]
                pts_pred = np.flip(pts_pred, axis=1)

                print('\t', class_names[cat], old_num_blobs == len(pts_pred), old_num_blobs, 'x', len(pts_pred), '->', len(pts_gt))

                if show:
                    ax.scatter(pts_gt[:, 0], pts_gt[:, 1], marker='o')
                    ax.scatter(pts_pred[:, 0], pts_pred[:, 1], marker='x')

                mse = mean_square_error(pts_gt, pts_pred) / config.scale
                ax.set_title('{}: {:.2e}'.format(class_names[cat], mse))
                mse_cat_dict[class_names[cat]] = mse
                mse_cat_list.append(mse)
                count_error_categories.append(pts_pred_center.shape[0] - pts_gt.shape[0])

            print(np.sum(old_blobs_counts), num_blobs, len(file_labels))

        """ Approach 1: """
        if False:
            for cat in range(pred.shape[2]):
                if show:
                    ax = axes[cat // 3, cat % 3]
                    ax.set_title(class_names[cat])
                preds_cat = pred[:, :, cat]

                if cat == 0:
                    # show prediction and skip blob detection
                    if show:
                        ax.imshow(preds_cat)
                    continue

                # prediction thresholding
                preds_cat[preds_cat < prediction_threshold] = 0.0
                preds_cat[preds_cat >= prediction_threshold] = 1.0
                # blob detection (connected components)
                blobs, num_blobs = skimage_label(preds_cat, return_num=True, connectivity=2)
                if num_blobs == 0:
                    if show:
                        ax.imshow(preds_cat)
                    continue

                pts_pred = []
                for blob in range(1, num_blobs + 1):
                    blob_indices = np.flip(np.argwhere(blobs == blob), axis=1)
                    center = blob_indices.mean(axis=0).astype(np.int)
                    pts_pred.append(center)

                pts_pred = np.vstack(pts_pred)
                pts_gt = file_labels[file_labels.category == class_names[cat]][['x', 'y']].to_numpy()
                if show:
                    # prediction blobs
                    ax.imshow(blobs)
                    # ground-truth points
                    gts = ax.scatter(pts_gt[:, 0], pts_gt[:, 1], marker='o')
                    # prediction centers
                    ps = ax.scatter(pts_pred[:, 0], pts_pred[:, 1], marker='x')

                mse = mean_square_error(pts_gt, pts_pred) / config.scale
                ax.set_title('{}: {:.2e}'.format(class_names[cat], mse))
                mse_cat_dict[class_names[cat]] = mse
                mse_cat_list.append(mse)

        # print("{:.2e}".format(mse_value))
        # print(file)
        # for cat, val in mse_categories.items():
        #     print('\t{}: {:.2e}'.format(cat, val))
        mse_file_total = np.mean(mse_cat_list)
        print(img_file_name, '{:.2e}'.format(mse_file_total))
        mse_totals.append(mse_file_total)
        if show:
            fig.legend(['prediction', 'ground-truth'])

            # original image
            img_path = os.path.join(images_dir, img_file_name)
            if os.path.isfile(img_path):
                img, _ = load_image(img_path, config.scale, config.center_crop_fraction)
                img, _ = crop_to_prediction(img, pred.shape)
                axes[2, 2].imshow(img)
                axes[2, 2].scatter(pts_pred_center[:, 0], pts_pred_center[:, 1])
            else:
                print('not found:', img_path)

            fig.suptitle(img_file_name)
            fig.tight_layout()
            fig.show()

        if output_location:
            fig.savefig(os.path.join(predictions_dir, img_file_name + '_plot_mse.png'), bbox_inches='tight')

    print('Total mean MSE:', np.mean(mse_totals))
    print('Total count MAE:', np.mean(count_error_categories))


    sns.histplot(blob_sizes, binwidth=5)
    plt.xlim(0, 400)
    plt.show()

    if False:
        f, ax = plt.subplots(1, 2, figsize=(18, 8))

        im = ax[0].imshow(preds_cat)
        ax[1].imshow(blobs, cmap='nipy_spectral')
        cbar = f.colorbar(im, ax=ax[1])
        plt.tight_layout()
        plt.show()

        pts_gt = np.array([np.array([x, x]) for x in np.arange(0, 500, 50)])
        pts_pred = np.array([pts_pred[0] * np.random.rand(*pts_pred[0].shape) for i in range(3)])
        plt.scatter(pts_gt_[:, 0], pts_gt_[:, 1], marker='x')
        plt.scatter(pts_pred_[:, 0], pts_pred_[:, 1], marker='+')
        plt.show()
