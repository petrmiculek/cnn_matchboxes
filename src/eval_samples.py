# stdlib
import os

# external
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
        fig_cm.figure.savefig(os.path.join(output_location, 'confusion_matrix' + '_val' * val + '.svg'),
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
        # false predictions == []
        print(err)
        accuracy = 100.0

    print('\tAccuracy: {0:0.3g}%'.format(accuracy))
    print(classification_report(labels, predictions))

    return accuracy


