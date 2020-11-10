import os

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def visualize_results(val_ds, model, save_outputs, class_names):
    """Visualize wrong predictions"""
    # separate dataset into images and labels
    # imgs_per_batch = np.array([list(x[0].numpy()) for x in list(val_ds)], dtype='object')
    # labels_per_batch = np.array([x[1].numpy() for x in list(val_ds)], dtype='object')

    # # only take first batch
    # imgs = np.array(imgs_per_batch[0])
    # labels = np.array(labels_per_batch[0])

    show_misclassified_regions = False
    misclassified_counter = 0
    misclassified_folder = 'missclassified_regions_{}_e{}'.format(model.name, len(model.losses))
    if save_outputs:
        os.makedirs(misclassified_folder, exist_ok=True)
    imgs = None
    labels = None

    for batch in list(val_ds):
        if imgs is None:
            imgs = np.array(list(batch[0].numpy()), dtype='object')
            labels = np.array(list(batch[1].numpy()), dtype='object')
        else:
            imgs = np.append(imgs, np.array(list(batch[0].numpy()), dtype='object'), axis=0)
            labels = np.append(labels, np.array(list(batch[1].numpy()), dtype='object'), axis=0)

    # predict and check predictions
    predictions_raw = model.predict(tf.convert_to_tensor(imgs, dtype=tf.float32))

    predictions = np.argmax(predictions_raw, axis=1)
    false_pred = np.where(labels != predictions)[0]  # reduce dimensions of a nested array

    # show misclassified images
    for idx in false_pred:
        label_true = class_names[labels[idx]]
        label_predicted = class_names[predictions[idx]]

        fig = plt.imshow(imgs[idx].astype("uint8"))

        # True label x Predicted label
        fig.axes.set_title('T:{} x F:{}'.format(label_true, label_predicted))
        fig.axes.axis("off")
        # fig.axes.figure.tight_layout(pad=1.0)

        if save_outputs:
            plot_path = os.path.join(misclassified_folder,
                                     '{}_{}_x_{}'.format(misclassified_counter, label_true, label_predicted))
            fig.savefig(plot_path, bbox_inches='tight')
        if show_misclassified_regions:
            fig.axes.figure.show()
        misclassified_counter += 1

    # confusion matrix
    conf_mat = confusion_matrix(list(labels), list(predictions))

    fig_cm = sns.heatmap(
        conf_mat, annot=True,
        xticklabels=class_names,
        yticklabels=class_names)
    fig_cm.set_title('Confusion Matrix\n{}'.format(model.name))
    fig_cm.set_xlabel("Predicted")
    fig_cm.set_ylabel("True")
    fig_cm.figure.tight_layout(pad=0.7)

    fig_cm.figure.show()
    if save_outputs:
        fig.savefig(os.path.join(misclassified_folder, 'confusion_matrix.png'), bbox_inches='tight')
