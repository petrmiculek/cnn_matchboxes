import functools
import itertools
import operator
import os
import numpy as np
import seaborn as sns
import tensorflow as tf
import cv2 as cv
import PIL

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as conf_mat
from math import ceil
from scipy.spatial.distance import cdist
from src_util.labels import load_labels
from math import log10, sqrt, floor
from src_util.general import safestr
from copy import deepcopy


def confusion_matrix(model_name, class_names, epochs_trained, labels,
                     predictions, output_location=None, show=True,
                     val=False, normalize=True):
    """Create and show/save confusion matrix"""
    kwargs = {}
    if normalize:
        # {'true', 'pred', 'all'}, default = None
        normalize = 'true'
        kwargs['min'] = 0.0
        kwargs['max'] = 0.0
        kwargs['fmt'] = '0.2f'
    else:
        normalize = None
        kwargs['fmt'] = 'd'

    cm = conf_mat(list(labels), list(predictions), normalize=normalize)

    fig_cm = sns.heatmap(
        cm,
        annot=True,
        xticklabels=class_names,
        yticklabels=class_names,
        # fmt='0.2f',
        # vmin=0.0,
        # vmax=1.0
        **kwargs
    )
    fig_cm.set_title('Confusion Matrix\n{} {} [e{}]'.format(model_name, 'val' if val else 'train', epochs_trained))
    fig_cm.set_xlabel("Predicted")
    fig_cm.set_ylabel("True")
    fig_cm.axis("on")
    fig_cm.figure.tight_layout(pad=0.5)

    if show:
        fig_cm.figure.show()

    if output_location:
        fig_cm.figure.savefig(os.path.join(output_location, 'confusion_matrix' + '_val' * val + '.png'),
                              bbox_inches='tight')

    plt.close(fig_cm.figure)


def misclassified_regions(imgs, labels, class_names, predictions,
                          false_pred, output_location=None, show=True):
    """Show misclassified regions

    # todo alternative version with grid-like output
    """

    for i, idx in enumerate(false_pred):
        label_true = class_names[labels[idx]]
        label_predicted = class_names[predictions[idx]]

        fig = plt.imshow(imgs[idx].astype("uint8"))

        # True label x Predicted label
        fig.axes.set_title('T:{} x F:{}'.format(label_true, label_predicted))
        fig.axes.axis("off")

        if output_location:
            # noinspection PyUnboundLocalVariable
            fig_location = os.path.join(output_location, '{}_{}_x_{}'.format(i, label_true, label_predicted))

            # should not be needed as the dir is created in caller (visualize_results)
            # d = os.path.dirname(fig_location)
            # if d and not os.path.isdir(d):
            #     os.makedirs(d)

            fig.axes.figure.savefig(fig_location, bbox_inches='tight')

        if show:
            fig.axes.figure.show()

        plt.close(fig.axes.figure)


# noinspection PyUnreachableCode
def visualize_results(model, dataset, class_names, epochs_trained,
                      output_location=None, show=False, misclassified=False, val=False):
    """Show misclassified regions and confusion matrix

    Get predictions for whole dataset and manually evaluate the model predictions.
    Use evaluated results to save misclassified regions and to show a confusion matrix.

    :param val:
    :param model:
    :param dataset:
    :param class_names:
    :param epochs_trained:
    :param output_location:
    :param show:
    :param misclassified:
    """

    """Dataset processing"""
    imgs = []
    labels = []

    for batch in list(dataset):
        imgs.append(batch[0].numpy())
        labels.append(batch[1].numpy())

    imgs = np.vstack(imgs)  # -> 4D [img_count x width x height x channels]
    labels = np.hstack(labels)  # -> 1D [img_count]

    """Making predictions"""
    predictions_raw = model.predict(tf.convert_to_tensor(imgs, dtype=tf.uint8))
    predictions_juice = tf.squeeze(predictions_raw)
    predictions = tf.argmax(predictions_juice, axis=1)

    false_predictions = np.where(labels != predictions)[0]  # reduce dimensions of a nested array

    """Misclassified regions"""
    if misclassified:
        if output_location:
            misclassified_dir = os.path.join(output_location, 'missclassified_regions')
            os.makedirs(misclassified_dir, exist_ok=True)
        else:
            misclassified_dir = None

        misclassified_regions(imgs, labels, class_names, predictions,
                              false_predictions, misclassified_dir, show=False)

    """Confusion matrix"""
    confusion_matrix(model.name, class_names, epochs_trained, labels,
                     predictions, output_location=output_location, show=show, val=val, normalize=False)

    """More metrics"""
    maxes = np.max(predictions_juice, axis=1)
    confident = len(maxes[maxes > 0.9])
    undecided = len(maxes[maxes <= 0.125])

    accuracy = 100.0 * (1 - len(false_predictions) / len(predictions))
    print('Accuracy: {0:0.2g}%'.format(accuracy))
    print('Prediction types:\n',
           '\tConfident: {}\n'.format(confident),
           '\tUndecided: {}'.format(undecided))

    return accuracy

    """
    for i in undecided_idx:
        fig = plt.imshow(np.squeeze(imgs[i]).astype("uint8"))
        fig.axes.figure.show()
    """
    """
    # print all predictions
    for i, img in enumerate(predictions_juice.numpy()):
        print('{}: '.format(i), end='')
        print(["{0:0.2f}".format(k) for k in img])
    """


def get_image_as_batch(img_path, scale=1.0, center_crop_fraction=0.5):
    """Open and Process image

    :param center_crop_fraction:
    :param img_path:
    :param scale:
    :return:
    """
    img = cv.imread(img_path)
    h, w, _ = img.shape  # don't use after resizing
    low = (1 - center_crop_fraction) / 2
    high = (1 + center_crop_fraction) / 2

    img = img[int(h * low): int(h * high), int(w * low): int(w * high), :]  # cut out from center

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # reversed indices, OK
    img_batch = np.expand_dims(img, 0)
    return img_batch


def make_prediction(model, img, maxes_only=False, undecided_only=False):
    """Make

    :param model:
    :param img: img as batch
    :param maxes_only:
    :param undecided_only:
    :return:
    """

    predictions_raw = model.predict(tf.convert_to_tensor(img, dtype=tf.uint8))
    predictions = tf.squeeze(predictions_raw).numpy()

    if maxes_only and not undecided_only:
        maxes = np.zeros(predictions.shape)  # todo tf instead of np
        max_indexes = np.argmax(predictions, axis=-1)
        maxes[np.arange(predictions.shape[0])[:, None],  # in every row
              np.arange(predictions.shape[1]),  # in every column
              max_indexes] = 1  # the maximum-predicted category (=channel) is set to 1

        predictions = maxes

    if undecided_only and not maxes_only:
        decisive_values = np.max(predictions, axis=2)
        decisive_indices = np.where(decisive_values < 0.5, True, False)
        decisive_indices = np.stack((decisive_indices,) * 8, axis=-1)
        undecided = np.where(decisive_indices, predictions, 0)

        predictions = undecided

    return predictions


def predict_full_image(model, class_names, labels, img_path, output_location=None,
                       heatmap_alpha=0.6, show=True, maxes_only=False, undecided_only=False):
    """Show predicted heatmaps for full image

    :param labels:
    :param heatmap_alpha:
    :param model:
    :param class_names:
    :param img_path:
    :param output_location:
    :param show:
    :param maxes_only: Show only the top predicted category per point.
    :param undecided_only: Show only predictions that were
    """

    scale = 0.5

    img = get_image_as_batch(img_path, scale)

    predictions = make_prediction(model, img, maxes_only, undecided_only)
    img = img[0]  # remove batch dimension

    # Model prediction is "cropped", adjust image accordingly
    img = img[15:-16, 15:-16]  # warning, depends on model cutout size (only works for 32x)

    category_titles = class_names

    if undecided_only:
        plots_title = 'undecided'
    elif maxes_only:
        plots_title = 'maxes'
    else:
        losses_sum, category_losses = full_img_pred_error(predictions, img_path, img, labels, class_names)
        category_titles = category_losses
        plots_title = str(losses_sum // 1e6) + 'M'

    display_predictions(predictions, img, img_path, category_titles, plots_title,
                        heatmap_alpha, show=show, output_location=output_location)


def display_predictions(predictions, img, img_path, class_names, title='', heatmap_alpha=0.6, show=True,
                        output_location=None, superimpose=False):
    # Plot predictions as heatmaps superimposed on input image
    # predictions = np.square(predictions)  # todo note down comparison y/n
    predictions = np.uint8(255 * predictions)
    class_activations = []

    print(predictions.shape)
    for i in range(8):
        pred = np.stack((predictions[:, :, i],) * 3, axis=-1)  # 2d to grayscale (R=G=B)
        pred = cv.applyColorMap(pred, cv.COLORMAP_VIRIDIS)
        if superimpose:
            pred = cv.cvtColor(pred, cv.COLOR_BGR2RGB)
            pred = cv.addWeighted(pred, heatmap_alpha, img, 1 - heatmap_alpha, gamma=0)
        class_activations.append(pred)

    class_activations.append(img)
    subplot_titles = np.append(class_names, 'full-image')

    # Plot heatmaps
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 14))
    fig.suptitle('Heatmaps\n{}'.format(title))
    fig.subplots_adjust(right=0.85, left=0.05)
    for i in range(9):
        ax = axes[i // 3, i % 3].imshow(class_activations[i])
        axes[i // 3, i % 3].axis('off')
        axes[i // 3, i % 3].set_title(subplot_titles[i])

    fig.tight_layout()
    if show:
        fig.show()

    if output_location:
        os.makedirs(output_location, exist_ok=True)
        img_path_no_suffix = safestr(img_path[0:img_path.rfind('.')])
        fig_location = os.path.join(output_location, 'heatmap_{}.png'.format(img_path_no_suffix))
        fig.savefig(fig_location, bbox_inches='tight')

    plt.close(fig)


def show_layer_activations(model, data_augmentation, ds, class_names, show=True, output_location=None):
    """Predict single cutout and show network's layer activations

    Adapted from:
    https://towardsdatascience.com/feature-visualization-on-convolutional-neural-networks-keras-5561a116d1af

    :param output_location:
    :param show:
    :param data_augmentation:
    :param model:
    :param ds:
    :param class_names:
    :return:
    """

    if output_location:
        output_location = os.path.join(output_location, 'layer_activations')
        os.makedirs(output_location, exist_ok=True)

    batch, labels = list(ds)[0]  # first batch
    batch_img0 = tf.convert_to_tensor(batch[0:1])  # first image (made to a 1-element batch)

    layers = [layer.output for layer in model.layers]
    model_all_outputs = tf.keras.Model(inputs=model.input, outputs=layers)

    # no augmentation, only crop
    batch_img0 = data_augmentation(batch_img0, training=False)

    print('GT   =', class_names[int(labels[0].numpy())])
    all_layer_activations = model_all_outputs(batch_img0, training=False)
    pred = all_layer_activations[-1].numpy()
    predicted_category = class_names[np.argmax(pred)]
    print('pred =', pred.shape, predicted_category)

    # I could test for region not being background here
    # ... or just read in images of known categories

    """ Show input image """
    fig_input, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(batch_img0[0].numpy().astype('uint8'), cmap='viridis', vmin=0, vmax=255)
    if output_location:
        plt.savefig(os.path.join(output_location, '00_input' + predicted_category))

    if show:
        plt.show()

    plt.close(fig_input.figure)

    layer_names = []
    for i, layer in enumerate(model.layers):
        layer_names.append(str(i) + '_' + layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, all_layer_activations):

        # debugging image scaling - how many values end up out of range
        out_of_range_count = 0
        total_count = 0

        # if 'batch_normalization' not in layer_name and layer_name not in layer_names[-3:-1]:
        #     continue

        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).

        # if size == 1:  # caused top == bottom range problems
        #     print(f'{layer_name} {n_features=}')
        #     continue

        n_cols = ceil(n_features / images_per_row)  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(min(images_per_row, n_features)):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]

                # don't normalize so as to keep proportions
                # channel_image -= np.mean(channel_image)
                # channel_image /= np.std(channel_image)

                channel_image *= 64
                channel_image += 128

                # how often do we clip
                min_, max_ = np.min(channel_image), np.max(channel_image)
                if min_ < 0.0 or max_ > 255.0:
                    out_of_range_count += 1
                total_count += 1

                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        fig = plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis', vmin=0, vmax=255)

        if output_location:
            plt.savefig(os.path.join(output_location, layer_name))

        if show:
            plt.show()

        plt.close(fig.figure)

        # print(f'{layer_name}: {out_of_range_count=} / {total_count}')


def heatmaps_all(model, class_names, val=True,
                 undecided_only=False, maxes_only=False, output_location=None, show=True):
    labels_dir = 'sirky' + '_val' * val
    labels = load_labels(labels_dir + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)

    if output_location:
        output_location = os.path.join(output_location, 'heatmaps')
        os.makedirs(output_location, exist_ok=True)

    for file in list(labels):
        predict_full_image(model, class_names,
                           labels,
                           img_path=labels_dir + os.sep + file,
                           output_location=output_location,
                           show=show,
                           undecided_only=undecided_only,
                           maxes_only=maxes_only,
                           )


def full_img_pred_error(predictions, img_path, img, labels, class_names):
    """full prediction metric

    todo work with prediction probability instead of argmax?

    non-background: nejbližší keypoint dané třídy -> dist == loss

    background: 0.0, pokud je více než K pixelů daleko od libovolného nejbližšího keypointu
                jinak pevná penalty

    # unused
    # xxyy = np.mgrid[0:img.shape[1], 0:img.shape[0]]

    """
    min_dist_background = 5.0
    min_dist_background_penalty = 10.0
    no_such_cat_penalty = 250.0

    def rescale_file_labels(file_labels):
        # todo test function: show annotations on a scaled-down image
        new = dict()
        for cat, l in file_labels.items():
            new_l = []
            for item in l:
                new_l.append(((int(item[0]) // 2 - 504 - 15), (int(item[1]) // 2 - 378 - 15)))
            new[cat] = new_l

        return new

    # filter only labels for given image
    file_labels = deepcopy([val for filename, val in labels.items() if
                            filename in img_path])  # key in img_name, because img_name contains full path

    if len(file_labels) != 1:
        print('invalid labels for file {}\n{}'.format(img_path, file_labels), file=sys.stderr)
        return float('NaN')

    file_labels = file_labels[0]
    file_labels = rescale_file_labels(file_labels)
    file_labels_merged = np.vstack([v for v in file_labels.values()])

    # make prediction one-hot over channels
    p_argmax = np.argmax(predictions, axis=2)

    # grid for calculating distances
    grid = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    grid = np.vstack([grid])
    grid = np.transpose(grid, (1, 2, 0))

    category_losses = {}

    # backgroundop
    bg_mask = np.where(p_argmax == 0, True, False)  # 2D mask
    bg_list = grid[bg_mask]  # list of 2D coords
    uniq = np.unique(bg_list, axis=0)
    if len(uniq) != len(bg_list):
        print(f'uniq {len(uniq)} / {len(bg_list)}')

    per_pixel_distances = cdist(bg_list, file_labels_merged)
    per_pixel_loss = np.min(per_pixel_distances, axis=1)
    penalized = per_pixel_loss[per_pixel_loss < min_dist_background]
    category_losses['background'] = min_dist_background_penalty * len(penalized)

    # keypoint categories
    # ppl = []
    for i in range(1, len(class_names)):  # skip 0 == background
        # todo same fix as for background -- not needed as I don't use grid here
        cat = class_names[i]

        # filter predictions of given category
        cat_list = grid[p_argmax == i]
        # cat_mask = np.where(p_argmax == i, True, False)  # 2D mask
        # cat_grid = None  # todo ended here

        if cat not in file_labels:  # no GT labels of this category
            # predicted non-present category
            category_losses[cat] = len(cat_list) * no_such_cat_penalty
            print(f'{cat} not in {img_path} -> {category_losses[cat]}')
            continue

        # distance of each predicted point to each GT label
        per_pixel_distances = np.square(cdist(cat_list, np.array(file_labels[cat])))
        per_pixel_loss = np.min(per_pixel_distances, axis=1)
        # print(per_pixel_distances.shape, per_pixel_loss.shape)

        category_losses[cat] = np.sum(per_pixel_loss)
        # ppl.append(per_pixel_loss)

    loss_sum = np.sum(list(category_losses.values()))

    return loss_sum,\
        [str(cat) + ': 1e{0:0.2g}'.format(log10(loss + 1)) for cat, loss in category_losses.items()]


def error_location(predictions, img, file_labels, class_names):
    """Attribute full image error in full image predictions

    todo calling of this

    :param predictions:
    :param img:
    :param file_labels:
    :param class_names:
    :return:

    inspiration (N-th answer):
    https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy
    """
    def dst_pt2grid(pt):
        return np.hypot(gx - pt[0], gy - pt[1])

    gx, gy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    flat_penalty = np.zeros((img.shape[0], img.shape[1])) + 0

    cat_distances = []
    for i, cat in enumerate(class_names):
        if cat not in file_labels:
            if cat == 'background':
                cat_distances.append(np.zeros((img.shape[0], img.shape[1])))
            else:
                cat_distances.append(flat_penalty)
            continue

        cat_labels = file_labels[cat]
        distance_layers = list(map(dst_pt2grid, cat_labels))
        distance_layers = np.vstack([distance_layers])
        min_distance = np.min(distance_layers, axis=0)  # minimum distance to any keypoint
        cat_distances.append(min_distance)

    cat_distances = np.dstack(cat_distances)
    titles = np.append(class_names, 'full_image')
    # cat_distances = cat_distances / np.max(cat_distances)

    # elementwise multiply predictions with cat_distances
    tst = predictions * cat_distances

    tst /= np.max(tst)

    display_predictions(tst, img, img_path, titles)
    # grid distance to keypoint can be cached and loaded from a pickle?


def show_augmentation(data_augmentation, dataset):
    """Show grid of augmentation results

    :param data_augmentation:
    :param dataset:
    :return:
    """

    """Convert dataset"""
    imgs = [img
            for batch in list(dataset)
            for img in batch[0]]

    # replaced by comprehension above
    # for batch in list(dataset):
    #     imgs.append(batch[0].numpy())

    imgs = np.vstack([imgs])  # -> 4D [img_count x width x height x channels]

    """Make predictions"""
    pred = data_augmentation(tf.convert_to_tensor(imgs, dtype=tf.uint8), training=True)

    half_dim = imgs.shape[2] // 2

    # Center-crop regions (64x to 32x)
    imgs = imgs[:, half_dim - 16: half_dim + 16, half_dim - 16: half_dim + 16, :]

    # alternatively, resize to 1/2 inside the loop below, once merged (B, W, H, C) -> (B*W, H, C)
    # imgs_col = cv.resize(imgs_col, None, fx=0.5, fy=0.5)

    """Show predictions as a grid"""
    rows = floor(sqrt(len(imgs)))  # Note: does not use all images
    cols = []

    for i in range(len(imgs) // rows):
        imgs_col = np.vstack(imgs[rows * i:(i + 1) * rows])
        pred_col = np.vstack(pred[rows * i:(i + 1) * rows])
        col = np.hstack((imgs_col, pred_col))
        cols.append(col)

    grid_img = np.hstack(cols)

    pi = PIL.Image.fromarray(grid_img.astype('uint8'))
    pi.show()

    # figure makes images too small
    # fig = plt.imshow(grid_img.astype("uint8"))
    # fig.axes.figure.show()
