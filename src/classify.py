# -*- coding: utf-8 -*-
"""
https://colab.research.google.com/drive/1F28FEGGLmy8-jW9IaOo60InR9VQtPbmG

todo:
fix annotation sirky (1 img)

21. 12.
zkus znovu projit tutorial na fine-tuning, pripadne grad cam



"""
import os
import sys

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import datetime

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from datasets import get_dataset
from show_results import visualize_results
from class_activation_map import single_image
import models

from IPython.display import Image, display
import matplotlib.cm as cm

"""
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap
"""

print(f'{tf.__version__=}')

print(tf.config.list_physical_devices('GPU'))

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

# tf.debugging.experimental.enable_dump_debug_info(os.path.join(logs_folder, 'debug'), tensor_debug_mode="FULL_HEALTH",
#                                                  circular_buffer_size=-1)

""" Create/Load a model """
model = models.fully_conv(num_classes)

print(model.summary())

# sys.exit(0)

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

    # _ = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=1,
    #     callbacks=[tensorboard_callback],
    #     class_weight=class_weights
    # )
    epochs = 100

    # epochs_trained = 0
    # epochs_trained += epochs

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                             patience=10,
                                             restore_best_weights=True)
                   ],
        class_weight=class_weights
    )

    epochs_trained = len(history.epoch)
    """
    """

    """Save model weights"""
    if save_outputs:
        model.save(saved_model_path)

# false predictions + confusion map
visualize_results(train_ds, model, save_outputs, class_names, epochs_trained)

"""
TODO
přidat None, None, 3
mám openCV?
zobrazit samostatně načtený obrázek
zobrazit jeden kanál
"""
"""
# get prediction for full image
img_path = '20201020_121210.jpg'
full_path = 'sirky/' + img_path
# canvas = single_image(model, full_path)
scale = 0.5

img_cv = cv.imread(full_path)
img_cv = cv.resize(img_cv, (int(img_cv.shape[1] * scale), int(img_cv.shape[0] * scale)))  # reversed indices, OK
img_batch = np.expand_dims(img_cv, 0)

predictions_raw = model.predict(tf.convert_to_tensor(img_batch, dtype=tf.uint8))
# np.max(predictions_raw)
predictions = np.argmax(predictions_raw, axis=1)
print(predictions.shape)
"""

# cv.imshow('.', img_cv)
# # canvas_color = cv.applyColorMap(canvas[0], cv.COLORMAP_JET)
# canvas_b = np.stack((canvas[0],)*3, axis=-1)
#
# superimposed_img = canvas_b * img_cv
# superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
#
# save_path = "output/" + img_path
# superimposed_img.save(save_path)
#
# # Display Grad CAM
# display(Image(save_path))
# blend = cv.imread(save_path)


"""
použil jsem resnet,
accuracy 
e1: 0.76
e7: 0.9242
plateau pod 0.94

trénuje se jen 65k parametrů na spodku sítě

z resnetu as is vyleze (2, 2, 2048)

to asi protože původně bral obrázky 32x32, ale já mu cpu 64x64

do flatten se nejspíš prostě rozloží na 1x4 

a v poslední vrstvě se trénuje mapping na 1 label

... to mi ale retrospektivně připadá jako shit přístup


"""

"""
# https://keras.io/guides/transfer_learning/
resnet = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    # weights=None,
    input_shape=(64, 64, 3),
    pooling='avg',  # average pooling into single prediction
    classes=2)  # does not seem to have any meaning
resnet.trainable = False

inputs = tf.keras.Input(shape=(64, 64, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = resnet(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
# x = tf.keras.layers.Flatten()(x)
# outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
# outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs, x)
"""
