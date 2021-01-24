import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2 as cv


def visualize_results(val_ds, model, save_outputs, class_names, epochs_trained):
    show_misclassified_regions = False

    misclassified_folder = 'missclassified_regions_{}_e{}'.format(model.name, len(model.losses))
    if save_outputs:
        os.makedirs(misclassified_folder, exist_ok=True)

    # split validation dataset -> images, labels
    imgs = []
    labels = []

    for batch in list(val_ds):
        imgs.append(batch[0].numpy())
        labels.append(batch[1].numpy())

    imgs = np.vstack(imgs)  # -> 4D [img_count x width x height x channels]
    labels = np.hstack(labels)  # -> 1D [img_count]

    predictions_raw = model.predict(tf.convert_to_tensor(imgs, dtype=tf.uint8))
    predictions_juice = tf.squeeze(predictions_raw)  # squeeze potentially more dimensions
    predictions = tf.argmax(predictions_juice, axis=1)

    false_pred = np.where(labels != predictions)[0]  # reduce dimensions of a nested array

    """Show misclassified regions"""
    for idx in false_pred:
        label_true = class_names[labels[idx]]
        label_predicted = class_names[predictions[idx]]

        fig = plt.imshow(imgs[idx].astype("uint8"))

        # True label x Predicted label
        fig.axes.set_title('T:{} x F:{}'.format(label_true, label_predicted))
        fig.axes.axis("off")

        if save_outputs:
            fig_location = os.path.join(misclassified_folder,
                                        '{}_{}_x_{}'.format(misclassified_counter, label_true, label_predicted))
            d = os.path.dirname(fig_location)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            fig.savefig(fig_location, bbox_inches='tight')

        if show_misclassified_regions:
            fig.axes.figure.show()

    """Confusion matrix"""
    conf_mat = confusion_matrix(list(labels), list(predictions))

    fig_cm = sns.heatmap(
        conf_mat,
        annot=True,
        xticklabels=class_names,
        yticklabels=class_names,
        fmt='d')
    fig_cm.set_title('Confusion Matrix\n{}[e{}]'.format(model.name, epochs_trained))
    fig_cm.set_xlabel("Predicted")
    fig_cm.set_ylabel("True")
    fig_cm.axis("on")
    fig_cm.figure.tight_layout(pad=0.5)
    fig_cm.figure.show()

    if save_outputs:
        fig_cm.savefig(os.path.join(misclassified_folder, 'confusion_matrix.png'), bbox_inches='tight')

    """More metrics"""
    maxes = np.max(predictions_juice, axis=1)
    confident = len(maxes[maxes > 0.9])
    undecided = len(maxes[maxes <= 0.125])
    # undecided_idx = np.argwhere(maxes <= 0.125)

    # print(labels.shape, predictions.shape)  # debug
    print('Accuracy:', 100.0 * (1 - len(false_pred) / len(predictions)), '%')
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


def predict_full_image(model, class_names, img_path, output_location, show_figure=True):
    """Show predicted heatmaps for full image"""
    scale = 0.5

    # Open and Process image
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))  # reversed indices, OK
    img_batch = np.expand_dims(img, 0)

    # Make prediction
    predictions_raw = model.predict(tf.convert_to_tensor(img_batch, dtype=tf.uint8))
    predictions = tf.squeeze(predictions_raw).numpy()
    # predictions_maxes = np.argmax(predictions_raw, axis=-1)

    class_activations = []
    heatmap_alpha = 0.5  # opacity

    # Turn predictions into heatmaps superimposed on input image
    predictions = np.uint8(255 * predictions)
    predictions = cv.resize(predictions, (img.shape[1], img.shape[0]))  # extrapolate predictions

    for i in range(0, len(class_names)):
        pred = np.stack((predictions[:, :, i],) * 3, axis=-1)
        pred = cv.applyColorMap(pred, cv.COLORMAP_HOT)  # COLORMAP_VIRIDIS
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
