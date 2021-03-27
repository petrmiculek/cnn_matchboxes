import tensorflow as tf


def mine_hard_cases(base_model, dataset):
    """

    todo test

    :param base_model:
    :param dataset:
    :return:
    """

    false_preds, imgs, labels, *_ = predict_all_tf(base_model, dataset)

    hard_ds = tf.data.Dataset.from_tensor_slices((tf.gather_nd(imgs, false_preds),
                                                  tf.gather_nd(labels, false_preds)))
    return hard_ds


def between_keypoints():
    import numpy as np
    import pandas as pd
    from labels import load_labels_pandas

    labels = load_labels_pandas('sirky_val/labels.csv', False, True)

    for img in labels.image.unique():
        print(img)
        file_labels = labels[labels.image == img]
    # todo continue here
