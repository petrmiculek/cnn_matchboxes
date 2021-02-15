import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as conf_mat
import cv2 as cv


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
        fig_cm.figure.savefig(os.path.join(output_location, 'confusion_matrix.png'), bbox_inches='tight')

    plt.close(fig_cm.figure)


def misclassified_regions(imgs, labels, class_names, predictions,
                          false_pred, misclassified_folder, show_misclassified):
    """Show misclassified regions"""
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

        if show_misclassified:
            fig.axes.figure.show()

        plt.close(fig.axes.figure)


def visualize_results(model, dataset, class_names, epochs_trained, 
                      output_location=None, show_figure=False, show_misclassified=False):
    """Show misclassified regions and confusion matrix

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
    predictions_juice = tf.squeeze(predictions_raw)  # squeeze potentially more dimensions
    predictions = tf.argmax(predictions_juice, axis=1)

    false_predictions = np.where(labels != predictions)[0]  # reduce dimensions of a nested array

    """Misclassified regions"""
    if output_location:
        misclassified_folder = output_location + os.sep + 'missclassified_regions_{}_e{}'.format(model.name, epochs_trained)
        os.makedirs(misclassified_folder, exist_ok=True)
    else:
        misclassified_folder = None

    if show_misclassified:
        misclassified_regions(imgs, labels, class_names, predictions,
                              false_predictions, misclassified_folder, show_misclassified)

    """Confusion matrix"""
    confusion_matrix(model, class_names, epochs_trained, labels,
                     predictions, output_location, show_figure)

    """More metrics"""
    maxes = np.max(predictions_juice, axis=1)
    confident = len(maxes[maxes > 0.9])
    undecided = len(maxes[maxes <= 0.125])
    # undecided_idx = np.argwhere(maxes <= 0.125)

    # print(labels.shape, predictions.shape)  # debug

    print('Accuracy:', 100.0 * (1 - len(false_predictions) / len(predictions)), '%')
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


def predict_full_image(model, class_names, img_path, output_location, 
                       heatmap_alpha=0.7, show_figure=True, maxes_only=False):
    """Show predicted heatmaps for full image

    :param model:
    :param class_names:
    :param img_path:
    :param output_location:
    :param show_figure:
    :param maxes_only: Show only the top predicted category per point.
    """

    scale = 0.5

    # Open and Process image
    img = cv.imread(img_path)
    h, w, _ = img.shape  # don't use after resizing
    img = img[h // 4: 3*h//4, w // 4: 3*w // 4, :]  # cut out half-sized from center
    # img = img[h // 3: 2 * h // 3, w // 3: 2 * w // 3, :]  # cut out third-sized from center
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # reversed indices, OK
    img_batch = np.expand_dims(img, 0)

    # Make prediction
    predictions_raw = model.predict(tf.convert_to_tensor(img_batch, dtype=tf.uint8))
    predictions = tf.squeeze(predictions_raw).numpy()

    # Model crops the image, accounting for it
    img = img[15:-16, 15:-16]

    if maxes_only:
        maxes = np.zeros(predictions.shape)
        max_indexes = np.argmax(predictions, axis=-1)
        maxes[np.arange(predictions.shape[0])[:, None], np.arange(predictions.shape[1]), max_indexes] = 1
        predictions = maxes

    class_activations = []

    # Turn predictions into heatmaps superimposed on input image
    predictions = np.uint8(255 * predictions)
    predictions = cv.resize(predictions, (img.shape[1], img.shape[0]))  # extrapolate predictions

    for i in range(0, len(class_names)):
        pred = np.stack((predictions[:, :, i],) * 3, axis=-1)
        pred = cv.applyColorMap(pred, cv.COLORMAP_VIRIDIS)  # COLORMAP_VIRIDIS
        pred = cv.cvtColor(pred, cv.COLOR_BGR2RGB)
        pred = cv.addWeighted(pred, heatmap_alpha, img, 1 - heatmap_alpha, gamma=0)
        class_activations.append(pred)

    class_activations.append(img)
    subplot_titles = np.append(class_names, 'full-image')

    # Plot heatmaps
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), )
    fig.suptitle('Class Activations Map')
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
