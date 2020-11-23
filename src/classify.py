# -*- coding: utf-8 -*-
"""
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG

todo:
fix annotation sirky (1 img)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import datetime
import models
from datasets import get_dataset
from show_results import visualize_results

print(f'{tf.__version__=}')

# from google.colab import drive
# drive.mount('/content/drive')
# data_dir = '/content/drive/My Drive/sirky/image_regions_64_050'

# save model weights, plotted imgs/charts
save_outputs = False

""" Load dataset """
data_dir = 'image_regions_64_050'

class_names, train_ds, val_ds, val_ds_batch, class_weights = get_dataset(data_dir)
num_classes = len(class_names)

""" Logging """
logs_folder = 'logs'

os.makedirs(logs_folder, exist_ok=True)

logdir = os.path.join(logs_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

tf.debugging.experimental.enable_dump_debug_info(os.path.join(logs_folder, 'debug'), tensor_debug_mode="FULL_HEALTH",
                                                 circular_buffer_size=-1)

""" Create/Load a model """
model = models.conv_tutorial(num_classes)

saved_model_path = os.path.join('models_saved', model.name)

# Load saved model
load_module = False

if load_module:
    model = tf.keras.models.load_model(saved_model_path)
    epochs_trained = 0
else:
    """ Train the model"""
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # only 1 epoch so that I can show model summary
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        callbacks=[tensorboard_callback],
        class_weight=class_weights
    )
    epochs_trained = 1

    print(model.summary())

    epochs = 100
    epochs_trained += epochs
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[tensorboard_callback],
        class_weight=class_weights
    )

    """Save model weights"""
    if save_outputs:
        model.save(saved_model_path)

visualize_results(val_ds, model, save_outputs, class_names, epochs_trained)
