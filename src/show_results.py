import functools
import itertools
import operator
import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as conf_mat
import cv2 as cv
from math import ceil
from scipy.spatial.distance import cdist
from src_util.labels import load_labels


def confusion_matrix(model, class_names, epochs_trained, labels,
                     predictions, output_location, show_figure):
    """Create and show/save confusion matrix"""
    cm = conf_mat(list(labels), list(predictions), normalize='true')

    fig_cm = sns.heatmap(
        cm,
        annot=True,
        xticklabels=class_names,
        yticklabels=class_names,
        fmt='0.2f',  # '0:0.2g'
        vmin=0.0,
        vmax=1.0
    )
    fig_cm.set_title('Confusion Matrix\n{}[e{}]'.format(model.name, epochs_trained))
    fig_cm.set_xlabel("Predicted")
    fig_cm.set_ylabel("True")
    fig_cm.axis("on")
    fig_cm.figure.tight_layout(pad=0.5)

    if show_figure:
        fig_cm.figure.show()

    if output_location:
        # todo fix file name to be unique
        fig_cm.figure.savefig(os.path.join(output_location, 'confusion_matrix' + model.name + '.png'),
                              bbox_inches='tight')

    plt.close(fig_cm.figure)


def misclassified_regions(imgs, labels, class_names, predictions,
                          false_pred, misclassified_folder, show_figure):
    """Show misclassified regions

    figures=images are saved when the misclassified_folder argument is given
    # todo alternative version with grid-like output
    """

    for i, idx in enumerate(false_pred):
        label_true = class_names[labels[idx]]
        label_predicted = class_names[predictions[idx]]

        fig = plt.imshow(imgs[idx].astype("uint8"))

        # True label x Predicted label
        fig.axes.set_title('T:{} x F:{}'.format(label_true, label_predicted))
        fig.axes.axis("off")

        if misclassified_folder:
            # noinspection PyUnboundLocalVariable
            fig_location = os.path.join(misclassified_folder,
                                        '{}_{}_x_{}'.format(i, label_true, label_predicted))
            d = os.path.dirname(fig_location)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            fig.axes.figure.savefig(fig_location, bbox_inches='tight')

        if show_figure:
            fig.axes.figure.show()

        plt.close(fig.axes.figure)


def visualize_results(model, dataset, class_names, epochs_trained,
                      output_location=None, show_figure=False, show_misclassified=False):
    """Show misclassified regions and confusion matrix

    Get predictions for whole dataset and manually evaluate the model predictions.
    Use evaluated results to save misclassified regions and to show a confusion matrix.

    :param model:
    :param dataset:
    :param class_names:
    :param epochs_trained:
    :param output_location:
    :param show_figure:
    :param show_misclassified:
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
    if output_location:
        misclassified_folder = output_location + os.sep + 'missclassified_regions_{}_e{}'.format(model.name,
                                                                                                 epochs_trained)
        os.makedirs(misclassified_folder, exist_ok=True)
    else:
        misclassified_folder = None

    if show_misclassified:
        misclassified_regions(imgs, labels, class_names, predictions,
                              false_predictions, misclassified_folder, show_figure=True)

    """Confusion matrix"""
    confusion_matrix(model, class_names, epochs_trained, labels,
                     predictions, output_location, show_figure)

    """More metrics"""
    maxes = np.max(predictions_juice, axis=1)
    confident = len(maxes[maxes > 0.9])
    undecided = len(maxes[maxes <= 0.125])
    # undecided_idx = np.argwhere(maxes <= 0.125)

    # print(labels.shape, predictions.shape)  # debug

    print('Accuracy: {0:0.2g}%'.format(100.0 * (1 - len(false_predictions) / len(predictions))))
    print('Prediction types:\n',
          '\tConfident: {}\n'.format(confident),
          '\tUndecided: {}'.format(undecided))

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

    img = img[h * int(low): h * int(high), w * int(low): w*int(high), :]  # cut out half-sized from center
    # img = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4, :]  # cut out half-sized from center
    # img = img[h // 3: 2 * h // 3, w // 3: 2 * w // 3, :]  # cut out third-sized from center

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
        # add undecided predictions - per element [x, y, z])
        lower_bound_indices = np.where(predictions > 0.1, True, False)
        upper_bound_indices = np.where(predictions < 0.2, True, False)

        # subtract decisive predictions - per spatial position [x, y, :]
        decisive_values = np.max(predictions, axis=2)
        decisive_indices = np.where(decisive_values > 0.8, True, False)
        decisive_indices = np.stack((decisive_indices,) * 8, axis=-1)

        undecided = np.where(lower_bound_indices, 1, 0)
        undecided = np.where(upper_bound_indices, undecided, 0)
        undecided = np.where(decisive_indices, 0, undecided)

        predictions = undecided

    return predictions


def predict_full_image(model, class_names, labels, img_path, output_location,
                       heatmap_alpha=0.6, show_figure=True, maxes_only=False, undecided_only=False):
    """Show predicted heatmaps for full image

    :param labels:
    :param heatmap_alpha:
    :param model:
    :param class_names:
    :param img_path:
    :param output_location:
    :param show_figure:
    :param maxes_only: Show only the top predicted category per point.
    :param undecided_only: Show only predictions that were
    """

    scale = 0.5

    img = get_image_as_batch(img_path, scale)

    predictions = make_prediction(model, img, maxes_only, undecided_only)
    img = img[0]  # remove batch dimension

    # Model prediction is "cropped", adjust image accordingly
    img = img[15:-16, 15:-16]  # warning, depends on model cutout size (only works for 32x)

    if undecided_only:
        plots_title = 'undecided'
    elif maxes_only:
        plots_title = 'maxes'
    else:
        img_loss = full_img_pred_error(predictions, img_path, img, labels, class_names)
        plots_title = str(img_loss // 1e6) + 'M'

    display_predictions(predictions, img, img_path, class_names, model, plots_title,
                        heatmap_alpha, show_figure, output_location)


def display_predictions(predictions, img, img_path, class_names, model, title='',
                        heatmap_alpha=0.6, show_figure=True, output_location=None):

    # Turn predictions into heatmaps superimposed on input image
    predictions = np.uint8(255 * predictions)
    class_activations = []

    for i in range(0, len(class_names)):
        pred = np.stack((predictions[:, :, i],) * 3, axis=-1)
        pred = cv.applyColorMap(pred, cv.COLORMAP_VIRIDIS)
        pred = cv.cvtColor(pred, cv.COLOR_BGR2RGB)
        pred = cv.addWeighted(pred, heatmap_alpha, img, 1 - heatmap_alpha, gamma=0)
        class_activations.append(pred)

    class_activations.append(img)
    subplot_titles = np.append(class_names, 'full-image')

    # Plot heatmaps
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), )
    fig.suptitle('Heatmaps\n{}'.format(title))
    fig.subplots_adjust(right=0.85, left=0.05)
    for i in range(9):
        ax = axes[i // 3, i % 3].imshow(class_activations[i])
        axes[i // 3, i % 3].axis('off')
        axes[i // 3, i % 3].set_title(subplot_titles[i])

    fig.tight_layout()
    if show_figure:
        fig.show()

    if output_location:
        img_path_no_suffix = img_path[0:img_path.rfind('.')].replace('/', '_')

        fig_location = os.path.join(output_location, 'heatmap_{}_{}.png'.format(img_path_no_suffix, model.name))
        d = os.path.dirname(fig_location)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        fig.savefig(fig_location, bbox_inches='tight')

    plt.close(fig)


def show_layer_activations(model_orig, model_features, ds, class_names):
    """Predict single cutout and show network's layer activations

    Adapted from:
    https://towardsdatascience.com/feature-visualization-on-convolutional-neural-networks-keras-5561a116d1af

    todo broken as of introducing the crop

    :param model_orig:
    :param model_features:
    :param ds:
    :param class_names:
    :return:
    """
    batch, labels = list(ds)[0]  # first batch
    batch_img0 = tf.convert_to_tensor(batch[0:1])  # first image (made to a 1-element batch)

    plt.imshow(batch_img0[0].numpy().astype('uint8'), aspect='auto', cmap='viridis', vmin=0, vmax=255)
    plt.show()

    print('GT label =', class_names[int(labels[0].numpy())])
    all_layer_activations = model_features(batch_img0, training=False)
    print('pred     =', all_layer_activations[-1].numpy().shape)

    layer_names = []

    for layer in model_orig.layers[1].layers[:]:
        layer_names.append(layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, all_layer_activations):  # Displays the feature maps
        out_of_range_count = 0
        if 'batch_normalization' not in layer_name and layer_name not in layer_names[-3:-1]:
            continue

        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        if size == 1:  # caused top == bottom range problems
            print(f'{layer_name} {n_features=}')
            continue
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
                min_, max_ = np.min(channel_image), np.max(channel_image)
                if min_ < 0.0 or max_ > 255:
                    out_of_range_count += 1
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis', vmin=0, vmax=255)
        plt.show()
        print(f'{out_of_range_count=}')


def heatmaps_all(model, class_names, name, val=True):
    folder = 'sirky' + '_val' * val
    labels = load_labels(folder + os.sep + 'labels.csv', use_full_path=False, keep_bg=False)
    for file in list(labels):
        predict_full_image(model, class_names,
                           labels,
                           img_path=folder + os.sep + file,
                           output_location='heatmaps' + name,
                           # output_location=None,
                           show_figure=True,
                           # undecided_only=True,
                           # maxes_only=True,
                           )


def full_img_pred_error(prediction, img_path, img, labels, class_names):
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
    no_such_cat_penalty = 50.0

    def rescale_file_labels(labels):
        new = dict()
        for cat, l in labels.items():
            new_l = []
            for item in l:
                new_l.append(((int(item[0]) // 2 - 504 - 15), (int(item[1]) // 2 - 378 - 15)))
            new[cat] = new_l

        return new

    # filter only labels for given image
    file_labels = [val for filename, val in labels.items() if filename in img_path]  # key in img_name, because img_name contains full path

    if len(file_labels) != 1:
        return float('NaN')

    file_labels = file_labels[0]
    file_labels = rescale_file_labels(file_labels)
    file_labels_merged = functools.reduce(operator.iconcat, [v for v in file_labels.values()])

    # make prediction one-hot over channels
    p_argmax = np.argmax(prediction, axis=2)

    # grid for calculating distances
    xxyy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    xy = np.reshape(xxyy, (img.shape[0], img.shape[1], 2))

    # keypoint categories
    category_losses = []
    for i, c_name in enumerate(class_names[1:]):

        # filter predictions of given category
        cat_idx = xy[p_argmax == i]

        if c_name not in file_labels:  # no GT labels of this category
            if len(cat_idx) > 0:  # predicted non-present category
                category_losses.append(len(cat_idx) * no_such_cat_penalty)
            continue

        # distance of each predicted point to each GT label
        per_pixel_distances = cdist(cat_idx, np.array(file_labels[c_name]))
        per_pixel_loss = np.min(per_pixel_distances, axis=1)
        print(np.mean(per_pixel_loss))

        category_losses.append(np.sum(per_pixel_loss))

    # background
    cat_idx = xy[p_argmax == 0]
    per_pixel_distances = cdist(cat_idx, np.array(file_labels_merged))
    per_pixel_loss = np.min(per_pixel_distances, axis=1)
    penalized = per_pixel_loss[per_pixel_loss < min_dist_background]
    category_losses.append(min_dist_background_penalty * len(penalized))

    loss_sum = np.sum(category_losses)
    return loss_sum

