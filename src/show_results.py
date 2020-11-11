import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


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

    predictions_raw = model.predict(tf.convert_to_tensor(imgs, dtype=tf.float32))  # use uint8 instead?

    predictions = np.argmax(predictions_raw, axis=1)
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
            plot_path = os.path.join(misclassified_folder,
                                     '{}_{}_x_{}'.format(misclassified_counter, label_true, label_predicted))
            fig.savefig(plot_path, bbox_inches='tight')

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