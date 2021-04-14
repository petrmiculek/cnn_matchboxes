import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure

from src.show_results import display_predictions
import run_config
from src_util.labels import load_labels, rescale_labels

if __name__ == '__main__':
    run_config.center_crop_fraction = 0.5
    run_config.scale = 0.5
    run_config.train_dim = 64
    model_crop_delta = run_config.train_dim - 1  # using outputs from 64x model

    labels = load_labels('sirky/labels.csv', False, False)
    # def rescale_file_labels_pandas(file_labels, orig_img_size, scale, model_crop_delta, center_crop_fraction):

    labels = rescale_labels(labels, run_config.scale, model_crop_delta, run_config.center_crop_fraction)

    class_names = sorted(['background'] + pd.unique(labels.category))

    predictions = []
    file_names = []
    for file in os.listdir('preds'):
        pred = np.load(os.path.join('preds', file))
        predictions.append(pred)
        file = file[:file.rfind('.')]  # remove .npy

        file_names.append(file)
        file_labels = labels[labels.image == file]

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        for cat in range(pred.shape[2]):
            # p_c = preds_cat.copy()
            preds_cat = pred[:, :, cat]
            # if cat != 0:
            #     # prediction thresholding
            #     preds_cat[preds_cat <= 0.99] = 0
            #     preds_cat[preds_cat > 0.99] = 1

            # axes[cat // 3, cat % 3].imshow(pred[:, :, cat])

            if cat == 0:
                # skip background
                continue

            blobs, num_blobs = measure.label(preds_cat, return_num=True, connectivity=2)
            if num_blobs == 0:
                # raise ValueError()
                continue

            print(cat, num_blobs)
            centers = []
            for blob in range(1, num_blobs + 1):
                blob_indices = np.argwhere(blobs == blob)
                center = blob_indices.mean(axis=0).astype(np.int)
                centers.append(center)

            centers = np.vstack(centers)
            print(cat, centers)

            axes[cat // 3, cat % 3].imshow(blobs)

            # predictions
            axes[cat // 3, cat % 3].scatter(centers[:, 1], centers[:, 0], marker='+')

            # ground-truth
            pts_gt = file_labels[file_labels.category == class_names[cat]][['x', 'y']].to_numpy()  # [file_labels.category == cat]
            plt.scatter(centers[:, 0], centers[:, 1])

        # plt.imshow(pred[:,:,0])
        plt.tight_layout()
        plt.show()


        if False:
            f, ax = plt.subplots(1, 2, figsize=(18, 8))

            im = ax[0].imshow(preds_cat)
            ax[1].imshow(blobs, cmap='nipy_spectral')
            cbar = f.colorbar(im, ax=ax[1])
            plt.tight_layout()
            plt.show()

            plt.imshow(preds_cat)
            plt.show()

            # display_predictions(predictions, img, img_path, class_titles, title='', show=True,
            #                     output_location=None, superimpose=False)
