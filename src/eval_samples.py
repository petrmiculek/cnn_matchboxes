# stdlib
import os
from math import ceil, floor, sqrt

# external
import PIL
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as conf_mat, classification_report

# local
import config


def confusion_matrix(labels, predictions, output_location=None, show=True, val=False, normalize=True):
    """Create and show/save confusion matrix"""
    kwargs = {}
    if normalize:
        # {'true', 'pred', 'all'}, default = None
        normalize = 'true'
        kwargs['vmin'] = 0.0
        kwargs['vmax'] = 1.0
        kwargs['fmt'] = '0.2f'
    else:
        normalize = None
        kwargs['fmt'] = 'd'

    cm = conf_mat(list(labels), list(predictions), normalize=normalize)

    fig_cm = sns.heatmap(
        cm,
        annot=True,
        xticklabels=config.class_names,
        yticklabels=config.class_names,
        # fmt='0.2f',
        # vmin=0.0,
        # vmax=1.0
        **kwargs
    )
    fig_cm.set_title('Confusion Matrix\n{} {} [e{}]'
                     .format(config.model_name, 'val' if val else 'train', config.epochs_trained))
    fig_cm.set_xlabel('Predicted')
    fig_cm.set_ylabel('True')
    fig_cm.axis('on')
    fig_cm.figure.tight_layout(pad=0.5)

    if show:
        fig_cm.figure.show()

    if output_location:
        fig_cm.figure.savefig(os.path.join(output_location, 'confusion_matrix' + '_val' * val + '.png'),
                              bbox_inches='tight')

    plt.close(fig_cm.figure)


def misclassified_samples(imgs, labels, class_names, predictions,
                          false_pred, output_location=None, show=True):
    """Show misclassified samples

    unused
    """

    if len(false_pred) > 500:
        print('too many misclassified regions, refusing to generate')
        return

    for i, idx in enumerate(false_pred):
        label_true = class_names[labels[idx]]
        label_predicted = class_names[predictions[idx]]

        fig = plt.imshow(imgs[idx].numpy().astype("uint8"))

        # True label x Predicted label
        fig.axes.set_title('T:{} x F:{}'.format(label_true, label_predicted))
        fig.axes.axis("off")

        if output_location:
            # noinspection PyUnboundLocalVariable
            fig_location = os.path.join(output_location, '{}_{}_x_{}'.format(i, label_true, label_predicted))
            fig.axes.figure.savefig(fig_location, bbox_inches='tight')

        if show:
            fig.axes.figure.show()

        plt.close(fig.axes.figure)


def predict_all_tf(model, dataset):
    """Get predictions for the whole dataset, while keeping original order of labels

    just using model.predict(dataset) leaves us with no info about the GT labels


    :param model:
    :param dataset:
    :return:
    """
    # using model, not base model - as I need the downscale from aug
    # imgs = []
    labels = []
    predictions = []
    for batch in list(dataset):
        # imgs.append(batch[0])
        labels.append(batch[1])
        predictions.append(model(batch[0], training=False))

    # imgs = tf.concat(imgs, axis=0)  # -> 4D [img_count x width x height x channels]
    labels = tf.concat(labels, axis=0)  # -> 1D [img_count]
    predictions = tf.concat(predictions, axis=0)  # -> 1D [img_count]

    # ds_reconstructed = tf.data.Dataset.from_tensor_slices(imgs).batch(64).prefetch(buffer_size=2)
    # predictions = model.predict(ds_reconstructed)
    predictions = tf.squeeze(predictions)
    predictions = tf.argmax(predictions, axis=1)

    false_predictions = tf.squeeze(tf.where(labels != predictions))

    return predictions, labels, false_predictions  # imgs 2nd


def predict_all_numpy(model, dataset):
    """Get predictions for the whole dataset, while keeping original order of labels

    just using model.predict(dataset) leaves us with no info about the GT labels

    superseded by predict_all_tf

    :param model:
    :param dataset:
    :return:
    """

    imgs = []
    labels = []
    for batch in list(dataset):
        imgs.append(batch[0].numpy())
        labels.append(batch[1].numpy())

    imgs = np.vstack(imgs)  # -> 4D [img_count x width x height x channels]
    labels = np.hstack(labels)  # -> 1D [img_count]

    """Making predictions"""
    imgs_tensor = tf.convert_to_tensor(imgs, dtype=tf.uint8)
    ds_reconstructed = tf.data.Dataset.from_tensor_slices(imgs_tensor)\
        .batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    predictions = model.predict(ds_reconstructed)
    predictions = tf.argmax(tf.squeeze(predictions), axis=1)
    false_predictions = np.where(labels != predictions)[0]  # reduce dimensions of a nested array

    return predictions, imgs, labels, false_predictions


def evaluate_model(model, dataset, val=False, output_location=None, show=False, misclassified=False):
    """Show misclassified regions and confusion matrix

    Get predictions for whole dataset and manually evaluate the model predictions.
    Use evaluated results to save misclassified regions and to show a confusion matrix.

    :param model:
    :param dataset:
    :param val:
    :param output_location:
    :param show:
    :param misclassified:
    """

    """Dataset processing"""
    predictions, labels, false_predictions = predict_all_tf(model, dataset)

    """Misclassified regions"""
    # unused
    # if misclassified:
    #     if output_location:
    #         misclassified_dir = os.path.join(output_location, 'missclassified_regions')
    #         os.makedirs(misclassified_dir, exist_ok=True)
    #     else:
    #         misclassified_dir = None
    #
    #     misclassified_samples(imgs, labels, class_names, predictions,
    #                           false_predictions, misclassified_dir, show=False)

    """Confusion matrix"""
    confusion_matrix(labels, predictions, output_location=output_location, show=show, val=val, normalize=False)

    """More metrics"""
    print('Validation:' if val else 'Training:')
    try:
        accuracy = 100.0 * (1 - len(false_predictions) / len(predictions))
    except TypeError as err:
        # false predictions is a scalar
        print(err)
        print(f'{false_predictions=}')
        accuracy = 100.0

    print('\tAccuracy: {0:0.3g}%'.format(accuracy))
    print(classification_report(labels, predictions))

    return accuracy


def show_layer_activations(model, data_augmentation, ds, show=True, output_location=None):
    """Predict single cutout and show network's layer activations

    Adapted from:
    https://towardsdatascience.com/feature-visualization-on-convolutional-neural-networks-keras-5561a116d1af

    :param output_location:
    :param show:
    :param data_augmentation:
    :param model:
    :param ds:
    :return:
    """

    if output_location:
        output_location = os.path.join(output_location, 'layer_activations')
        os.makedirs(output_location, exist_ok=True)

    # choose non-background sample
    batch, labels = next(iter(ds))  # first batch
    idx = np.argmax(labels)

    batch_img0 = tf.convert_to_tensor(batch[idx: idx + 1])  # first image (made to a 1-element batch)

    layers = [layer.output for layer in model.layers]
    model_all_outputs = tf.keras.Model(inputs=model.input, outputs=layers)

    # no augmentation, only crop
    batch_img0 = data_augmentation(batch_img0, training=False)

    print('GT   =', config.class_names[int(labels[idx].numpy())])
    all_layer_activations = model_all_outputs(batch_img0, training=False)
    pred = all_layer_activations[-1].numpy()
    predicted_category = config.class_names[np.argmax(pred)]
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

        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).

        n_cols = ceil(n_features / images_per_row)  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(min(images_per_row, n_features)):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]

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

        scale = 1.0 / size
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


def show_augmentation(data_augmentation, dataset):
    """Show grid of augmentation results

    :param data_augmentation:
    :param dataset:
    :return:
    """

    """Convert dataset"""
    # todo use predict_all_tf, with added training param
    imgs = [img
            for batch in list(dataset)
            for img in batch[0]]

    imgs = np.vstack([imgs])  # -> 4D [img_count x width x height x channels]

    """Make predictions"""
    pred = data_augmentation(tf.convert_to_tensor(imgs, dtype=tf.uint8), training=True)

    low = 1 * imgs.shape[2] // 4
    high = 3 * imgs.shape[2] // 4

    # Center-crop regions (64x to 32x)
    imgs = imgs[:, low:high, low:high, :]

    # alternatively, resize to 1/2 inside the loop below, once merged (B, W, H, C) -> (B*W, H, C)
    # imgs_col = cv.resize(imgs_col, None, fx=0.5, fy=0.5)

    """Show predictions as a grid"""
    rows = floor(sqrt(len(imgs)))  # Note: does not use all images
    cols = []

    for i in range(len(imgs) // rows):
        # make a column [W, N * H]
        imgs_col = np.vstack(imgs[rows * i:(i + 1) * rows])
        pred_col = np.vstack(pred[rows * i:(i + 1) * rows])
        col = np.hstack((imgs_col, pred_col))
        cols.append(col)

    grid_img = np.hstack(cols)

    pi = PIL.Image.fromarray(grid_img.astype('uint8'))
    pi.show()
