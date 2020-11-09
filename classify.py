# -*- coding: utf-8 -*-
"""
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG
"""
import datetime
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import models
from sklearn.metrics import confusion_matrix
import tempfile
import tensorboard
import seaborn as sns
import glob
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f'{tf.__version__=}')

# from google.colab import drive
# drive.mount('/content/drive')
save_outputs = False

""" Load a dataset """

batch_size = 32
img_height = 64
img_width = 64
# data_dir = '/content/drive/My Drive/sirky/image_regions_64_050'
data_dir = 'image_regions_64_050'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

""" Visualize the data """

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#
# plt.show()

""" Configure dataset for performance

buffered prefetching = yield data without I/O blocking. 
two important methods when loading data.

`.cache()` keeps the images in memory after they're loaded off disk during the first epoch. 
This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, 
you can also use this method to create a performant on-disk cache.

`.prefetch()` overlaps data preprocessing and model execution while training. 

[data performance guide](https://www.tensorflow.org/guide/data_performance#prefetching).
"""

val_as_batch_dataset = val_ds

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


""" Logging """
logs_folder = 'logs'

os.makedirs(logs_folder, exist_ok=True)

logdir = os.path.join(logs_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

tf.debugging.experimental.enable_dump_debug_info(os.path.join(logs_folder, 'debug'), tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)


""" Create/Load a model """
model = models.conv_tutorial_tweaked(num_classes)

saved_model_path = os.path.join('model_saved_', model.name)

# Load saved model
load_module = False

if load_module:
    model = tf.keras.models.load_model(saved_model_path)
else:
    """ Train a model"""
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[tensorboard_callback]
    )

    """Save model weights"""
    if save_outputs:
        model.save(saved_model_path)

        # weights only -> cannot be trained after loading
        # model.save_weights(learned_weights)

"""Visualize wrong predictions"""

# separate dataset into images and labels
imgs_per_batch = np.array([list(x[0].numpy()) for x in list(val_ds)], dtype='object')
labels_per_batch = np.array([x[1].numpy() for x in list(val_ds)], dtype='object')
# # only take first batch
# imgs = np.array(imgs_per_batch[0])
# labels = np.array(labels_per_batch[0])

misclassified_counter = 0
if save_outputs:
    misclassified_folder = 'missclassified_regions_{}_e{}'.format(model.name, len(model.losses))
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

    # True label x Predicted label
    plt.title('T:{} x F:{}'.format(label_true, label_predicted))

    plt.axis("off")
    plt.imshow(imgs[idx].astype("uint8"))
    plt.tight_layout()

    if save_outputs:
        plot_path = os.path.join(misclassified_folder,
                                 '{}_{}_x_{}'.format(misclassified_counter, label_true, label_predicted))
        plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    misclassified_counter += 1

# confusion matrix
conf_mat = confusion_matrix(list(labels), list(predictions))

sns.heatmap(
    conf_mat, annot=True,
    xticklabels=class_names,
    yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()
if save_outputs:
    plt.savefig(os.path.join(misclassified_folder, 'confusion_matrix'), bbox_inches='tight')

"""

custom training loop instead of using `model.fit`. 
To learn more, visit https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

more about overfitting and how to reduce it in https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
"""

