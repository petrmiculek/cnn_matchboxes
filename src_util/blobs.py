# stdlib
import os

# external
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
# from skimage.io import imread
# from skimage.transform import resize

# local
# from src.show_results import display_predictions
import run_config
from src_util.labels import load_labels, rescale_labels
from src.eval_images import load_image, crop_to_prediction

if __name__ == '__main__':
    run_config.center_crop_fraction = 0.5
    run_config.scale = 0.5
    run_config.train_dim = 64
    model_crop_delta = run_config.train_dim - 1  # using outputs from 64x model

    predictions_dir = 'preds'
    images_dir = 'sirky'
    labels = load_labels(os.path.join(images_dir, 'labels.csv'), False, False)
    labels = rescale_labels(labels, run_config.scale, model_crop_delta, run_config.center_crop_fraction)
    class_names = ['background'] + sorted(pd.unique(labels.category))

    predictions = []
    file_names = []
    for file in os.listdir(predictions_dir):
        pred = np.load(os.path.join(predictions_dir, file))
        predictions.append(pred)
        file = file[:file.rfind('.')]  # remove .npy
        file_names.append(file)
        file_labels = labels[labels.image == file]

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        for cat in range(pred.shape[2]):
            ax = axes[cat // 3, cat % 3]
            ax.set_title(class_names[cat])
            preds_cat = pred[:, :, cat]
            # p_c = preds_cat.copy()
            if cat == 0:
                # skip background
                ax.imshow(preds_cat)
                continue

            # prediction thresholding
            preds_cat[preds_cat <= 0.99] = 0
            preds_cat[preds_cat > 0.99] = 1
            blobs, num_blobs = measure.label(preds_cat, return_num=True, connectivity=2)
            if num_blobs == 0:
                # raise ValueError()
                # ax.imshow(pred[:, :, cat])
                continue

            centers = []
            for blob in range(1, num_blobs + 1):
                blob_indices = np.argwhere(blobs == blob)
                center = blob_indices.mean(axis=0).astype(np.int)
                # center becomes [y, x], why
                centers.append(center)

            centers = np.vstack(centers)
            # prediction blobs
            ax.imshow(blobs)
            # ground-truth points
            pts_gt = file_labels[file_labels.category == class_names[cat]][['x', 'y']].to_numpy()
            gts = ax.scatter(pts_gt[:, 0], pts_gt[:, 1], marker='o')
            # prediction centers
            ps = ax.scatter(centers[:, 1], centers[:, 0], marker='x')

        fig.legend(['prediction', 'ground-truth'])

        img_path = os.path.join(images_dir, file)
        if os.path.isfile(img_path):
            img, _ = load_image(img_path, run_config.scale, run_config.center_crop_fraction)
            img, _ = crop_to_prediction(img, preds_cat.shape)
            axes[2, 2].imshow(img)
        else:
            print('not found:', img_path)

        fig.suptitle(file)
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(predictions_dir, file + '_plot.png'), bbox_inches='tight')

        if False:
            f, ax = plt.subplots(1, 2, figsize=(18, 8))

            im = ax[0].imshow(preds_cat)
            ax[1].imshow(blobs, cmap='nipy_spectral')
            cbar = f.colorbar(im, ax=ax[1])
            plt.tight_layout()
            plt.show()
