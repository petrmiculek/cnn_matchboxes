# stdlib
import os
import sys

curr_path = os.getcwd()
sys.path.extend([curr_path] + [d for d in os.listdir() if os.path.isdir(d)])

# external
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local
import config
import model_build
import models
from datasets import get_dataset
from eval_images import eval_full_predictions_all, prediction_to_keypoints, \
    make_prediction, load_image, crop_to_prediction
from eval_samples import evaluate_model
from display import display_predictions, display_keypoints, show_layer_activations
from counting import get_gt_points, count_crates

if __name__ == '__main__':
    config.train = False
    config.dataset_size = 1000
    config.center_crop_fraction = 1.0
    # depends on model
    # train dim decided by model
    config.scale = 0.25

    # should not matter
    config.augment = True

    config.output_location = 'eval_outputs'
    config.show = True

    load_model_name = '64x_d1-3-5-7-9-11-1-1_2021-05-10-05-53-28_full'  # datasets/128x_025s_1000bg

    config.model_name = load_model_name + '_reloaded'

    try:
        dim_str = load_model_name.split('x_')[0]
        dim_str = ''.join(c for c in dim_str if c.isdigit())
        dim = int(dim_str)
    except IndexError:
        print('', file=sys.stderr)
        dim = 64  # fallback

    config.train_dim = dim
    config.dataset_dim = 2 * dim

    base_model, model, aug_model = model_build.load_model(load_model_name, load_weights=True)
    config.epochs_trained = 123  # avoid epoch number confusion in tensorboard

    dataset_dir = 'datasets/{}x_{:03d}s_{}bg'\
        .format(config.dataset_dim, int(100 * config.scale), config.dataset_size)

    val_ds, _, _ = get_dataset(dataset_dir + '_val')
    train_ds, config.class_names, _ = get_dataset(dataset_dir)

    if False:
        # show augmented samples from dataset
        imgs, labels = next(iter(train_ds))
        idx = np.argwhere(labels.numpy() > 0)
        # rozšířit indexy
        idx = np.append(idx, np.arange(255 - len(idx), 255))
        imgs = imgs.numpy()[idx]
        labels = labels.numpy()[idx]
        imgs_ = imgs[::2]
        from src.display import show_augmentation

        show_augmentation(aug_model, imgs)

    if False:
        config.output_location = os.path.join('eval_outputs', 'svg_' + model.name)
        # run evaluation
        mse_pix_val, mse_val, keypoint_count_mae_val, crate_count_mae_val, crate_count_failrate_val = \
            eval_full_predictions_all(base_model, val=True, output_location=config.output_location, show=config.show)

        mse_pix_train, mse_train, keypoint_count_mae_train, crate_count_mae_train, crate_count_failrate_train = \
            eval_full_predictions_all(base_model, val=False, output_location=config.output_location, show=config.show)

        val_accu = evaluate_model(model, val_ds, val=True, output_location=config.output_location,
                                      show=config.show, misclassified=False)

        evaluate_model(model, train_ds, val=False, output_location=config.output_location, show=config.show)

        val_metrics = model.evaluate(val_ds, verbose=0)
        train_metrics = model.evaluate(train_ds, verbose=0)

        show_layer_activations(base_model, aug_model, val_ds, show=False,
                                       output_location=config.output_location)


    if False:
        # comparing predictions at multiple scales
        output_location = 'scale_comparison'
        os.makedirs(output_location, exist_ok=True)
        config.show = False
        show = False
        for s in np.arange(start=0.25, stop=1.26, step=0.125):
            print(s)

            config.center_crop_fraction = 0.2
            config.scale = s

            images_dir = 'sirky' + '_val' * val
            file = sorted(os.listdir(images_dir))[-1]

            for ccf in np.arange(0.10, 1.05, 0.05):
                config.center_crop_fraction = ccf
                print(ccf)

                labels = load_labels(images_dir + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)
                labels = resize_labels(labels, config.scale, config.train_dim - 1, config.center_crop_fraction)

                file_labels = labels[labels.image == file]
                pix_mse, point_mse, count_mae = \
                    full_prediction(base_model, file_labels, img_path=images_dir + os.sep + file,
                                    output_location=output_location, show=show)

    if False:
        show = True
        output_location = os.path.join('outputs', base_model.name, 'crate_count')
        os.makedirs(output_location, exist_ok=True)

        images_dir = 'sirky'  # + '_val'
        file = os.listdir(images_dir)[-3]

        counts_gt = pd.read_csv(os.path.join('sirky', 'count.txt'),
                                header=None,
                                names=['image', 'cnt'],
                                dtype={'image': str, 'cnt': np.int32})

        for file in os.listdir(images_dir):
            if not file.endswith('.jpg'):
                continue

            file = '20201020_115946.jpg'
            config.center_crop_fraction = 0.1

            img_path = os.path.join(images_dir, file)
            img, orig_size = load_image(img_path, 1.0, config.center_crop_fraction)

            prediction = make_prediction(base_model, img)
            img, crop_delta = crop_to_prediction(img, prediction.shape)
            keypoints, categories = prediction_to_keypoints(prediction)
            keypoints, categories = remove_keypoint_outliers(keypoints, categories)

            df = pd.DataFrame(np.c_[keypoints, categories], columns=['x', 'y', 'category'])
            label_dict = dict(enumerate(config.class_names))
            df['category'] = df['category'].map(label_dict)
            count_gt = np.array(counts_gt[counts_gt.image == file].cnt)[0]
            count_pred = count_crates(df)

            print(f'Count: prediction = {count_pred:0.2g}, gt = {count_gt:0.2g}')

            title = f'{file}\nPred: {count_pred:0.2g}, GT: {count_gt:0.2g}'
            display_keypoints((keypoints, categories), img, img_path, config.class_names, title=title,
                                show=show, output_location=output_location)

            from labels import load_labels, resize_labels
            labels = load_labels(images_dir + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)
            labels = resize_labels(labels, 1.0, 0, config.center_crop_fraction)
            file_labels = labels[labels.image == file]
            plt.imshow(img)
