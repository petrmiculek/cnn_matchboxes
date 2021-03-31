import os
import util

import tensorflow as tf
from tensorflow.keras.layers import \
    add, Conv2D, BatchNormalization, Softmax, \
    Input, MaxPool2D, Cropping2D, Concatenate, \
    AvgPool2D, ZeroPadding2D, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import \
    CenterCrop, RandomFlip, RandomRotation, Rescaling

from src_util.general import safestr


def augmentation(aug=True, crop_to=64, ds_dim=64):
    aug_model = tf.keras.Sequential(name='augmentation')
    aug_model.add(Input(shape=(ds_dim, ds_dim, 3)))

    if aug:
        aug_model.add(RandomFlip("horizontal"))
        aug_model.add(RandomRotation(1 / 16))  # =rot22.5°
        aug_model.add(util.RandomColorDistortion(brightness_delta=0.3,
                                                 contrast_range=(0.25, 1.25),
                                                 hue_delta=0.1,
                                                 saturation_range=(0.75, 1.25)))

    assert ds_dim >= crop_to
    if ds_dim != crop_to:
        aug_model.add(CenterCrop(crop_to, crop_to))

    if not aug and crop_to == ds_dim:
        # no other layers, model cannot be empty
        # base Layer class == identity layer
        aug_model.add(tf.keras.layers.Layer(name='identity'))

    return aug_model


def dilated_64x_exp2(num_classes, name_suffix='', **kwargs):
    """
    March 15

    Dilation rate growth accounts for receptive field growth caused by pooling

    :param num_classes:
    :param name_suffix:
    :return:
    """
    he_norm = tf.keras.initializers.he_normal()
    conv_args = {
        'activation': 'relu',
        'padding': 'valid',
        'kernel_initializer': he_norm
    }
    base_width = kwargs['base_width'] if 'base_width' in kwargs else 16

    x = Input(shape=(None, None, 3))  # None, None
    input_layer = x

    for i, width_coef in zip([1, 3, 5, 7, 9], [2, 4, 4, 8, 8]):
        w = base_width * width_coef
        x = Conv2D(w, 3, **conv_args, dilation_rate=i)(x)
        x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)

    x = Conv2D(16 * base_width, 2, **conv_args, dilation_rate=11)(x)  # -> 3x3

    x = BatchNormalization()(x)
    x = Conv2D(16 * base_width, 3, **conv_args)(x)  # fit-once

    x = BatchNormalization()(x)
    x = Conv2D(16 * base_width, 1, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm, activation=None)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='dilated_64x_exp2_' + name_suffix)
    return model, 64


def dilated_32x_exp2(num_classes, name_suffix='', **kwargs):
    """
    March 15

    Dilation rate growth accounts for receptive field growth caused by pooling

    :param num_classes:
    :param name_suffix:
    :return:
    """
    he_norm = tf.keras.initializers.he_normal()
    conv_args = {
        'activation': 'relu',
        'padding': 'valid',
        'kernel_initializer': he_norm
    }
    base_width = kwargs['base_width'] if 'base_width' in kwargs else 16
    tail_downscale = kwargs['tail_downscale'] if 'tail_downscale' in kwargs else 1

    x = Input(shape=(None, None, 3))  # None, None
    input_layer = x

    # non-cropping conv
    x = Conv2D(base_width, 3, kernel_initializer=he_norm, activation='relu', padding='same')(x)

    if kwargs['pool_after_first_conv']:
        x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)

    for i, width_coef in zip([3, 5, 7], [2, 4, 8 // tail_downscale]):
        x = BatchNormalization()(x)

        w = base_width * width_coef
        x = Conv2D(w, 3, **conv_args, dilation_rate=i)(x)

        x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(8 * base_width // tail_downscale, 2, **conv_args)(x)  # fit-once

    x = BatchNormalization()(x)
    x = Conv2D(8 * base_width // tail_downscale, 1, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm, activation=None)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='dilated_32x_exp2_' + name_suffix)
    return model, 32
