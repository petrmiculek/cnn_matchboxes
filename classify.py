# -*- coding: utf-8 -*-
"""
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG

todo:
why no labels in confusion matrix?
why is confusion matrix not shown when plotting misclassified regions?

"""
import datetime
import os
import tensorflow as tf
import models
from datasets import get_dataset
from show_results import visualize_results

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(f'{tf.__version__=}')

# from google.colab import drive
# drive.mount('/content/drive')

# model weights, plotted imgs/charts
save_outputs = False

""" Load a dataset """

# data_dir = '/content/drive/My Drive/sirky/image_regions_64_050'
data_dir = 'image_regions_64_050'

class_names, train_ds, val_ds, val_ds_batch = get_dataset(data_dir)
num_classes = len(class_names)

""" Visualize the data """
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

""" Logging """
logs_folder = 'logs'

os.makedirs(logs_folder, exist_ok=True)

logdir = os.path.join(logs_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

tf.debugging.experimental.enable_dump_debug_info(os.path.join(logs_folder, 'debug'), tensor_debug_mode="FULL_HEALTH",
                                                 circular_buffer_size=-1)

""" Create/Load a model """
model = models.conv_tutorial(num_classes)

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

    # only 1 epoch so that I can show model summary
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        callbacks=[tensorboard_callback]
    )

    # print(model.summary())

    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=20,
    #     callbacks=[tensorboard_callback]
    # )

    """Save model weights"""
    if save_outputs:
        model.save(saved_model_path)

        # weights only -> cannot be trained after loading
        # model.save_weights(learned_weights)

visualize_results(val_ds, model, save_outputs, class_names)

"""

custom training loop instead of using `model.fit`. 
To learn more, visit https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

more about overfitting and how to reduce it in https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
"""
