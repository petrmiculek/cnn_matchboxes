import os
from math import log2, ceil
import util

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import \
    add, Conv2D, BatchNormalization, Softmax, Input, MaxPool2D, Cropping2D, Concatenate, AvgPool2D, ZeroPadding2D
from tensorflow.keras.layers.experimental.preprocessing import \
    CenterCrop, RandomFlip, RandomRotation

from src_util.general import safestr


def augmentation(aug=True, crop_to=64, orig_size=64):
    aug_model = tf.keras.Sequential(name='augmentation')
    aug_model.add(Input(shape=(orig_size, orig_size, 3)))

    if aug:
        aug_model.add(RandomFlip("horizontal"))
        # aug_model.add(RandomRotation(1 / 16))  # =rot22.5°
        aug_model.add(util.RandomColorDistortion(brightness_delta=0.3,
                                                 contrast_range=(0.5, 1.5),
                                                 hue_delta=0.1,
                                                 saturation_range=(0.5, 1.5)))

    if crop_to != 64:
        aug_model.add(CenterCrop(crop_to, crop_to))

    if not aug and crop_to == 64:
        # no other layers, model cannot be empty
        # base Layer class == identity layer
        aug_model.add(tf.keras.layers.Layer(name='identity'))

    return aug_model


def dilated_64x_exp2(num_classes, name_suffix=''):
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
    base_width = 8  # can go 1 up

    x = Input(shape=(None, None, 3))  # None, None
    input_layer = x

    for i, width_coef in zip([1, 3, 5, 7, 9], [2, 4, 4, 8, 8]):
        w = base_width * width_coef
        x = Conv2D(w, 3, **conv_args, dilation_rate=i)(x)

        x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)

    x = BatchNormalization()(x)
    x = Conv2D(16 * base_width, 2, **conv_args, dilation_rate=11)(x)  # -> 3x3

    x = BatchNormalization()(x)
    x = Conv2D(16 * base_width, 3, **conv_args)(x)  # fit-once

    x = BatchNormalization()(x)
    x = Conv2D(32 * base_width, 1, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm, activation=None)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='dilated_64x_exp2_' + name_suffix)
    return model, 64


def dilated_32x_exp2_wider(num_classes, name_suffix=''):
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
    base_width = 64

    x = Input(shape=(None, None, 3))  # None, None
    input_layer = x

    # non-cropping conv
    x = Conv2D(base_width, 3, kernel_initializer=he_norm, activation='relu', padding='same')(x)

    for i, width_coef in zip([3, 5, 7], [2, 4, 8]):
        x = BatchNormalization()(x)

        w = base_width * width_coef
        x = Conv2D(w, 3, **conv_args, dilation_rate=i)(x)

        x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(16 * base_width, 2, **conv_args)(x)  # fit-once

    x = BatchNormalization()(x)
    x = Conv2D(16 * base_width, 1, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm, activation=None)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='dilated_32x_exp2_wider_' + name_suffix)
    return model, 32


def dilated_32x_exp2(num_classes, name_suffix=''):
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
        # 'padding': 'same',  # check
        'kernel_initializer': he_norm
    }
    base_width = 32

    x = Input(shape=(None, None, 3))  # None, None
    input_layer = x

    # non-cropping conv
    x = Conv2D(base_width, 3, kernel_initializer=he_norm, activation='relu', padding='same')(x)

    for i, width_coef in zip([3, 5, 7], [2, 4, 8]):
        x = BatchNormalization()(x)

        w = base_width * width_coef
        x = Conv2D(w, 3, **conv_args, dilation_rate=i)(x)

        x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(16 * base_width, 2, **conv_args)(x)  # fit-once

    x = BatchNormalization()(x)
    x = Conv2D(16 * base_width, 1, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm, activation=None)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='dilated_32x_exp2' + name_suffix)
    return model, 32


def residual_64x_33l_concat(num_classes, name_suffix=''):
    """
    March 15

    Dilation 2
    1x1 conv in common branch

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
    width = 64

    def residual_block(x):
        y = Cropping2D(cropping=((2, 2), (2, 2)), )(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, **conv_args, dilation_rate=2)(x)

        x = Concatenate()([x, y])
        x = Conv2D(width, 1, **conv_args)(x)
        return x

    x = Input(shape=(64, 64, 3))  # None, None
    input_layer = x

    x = Conv2D(width, 3, **conv_args)(x)

    for i in range(15):
        x = residual_block(x)

    x = BatchNormalization()(x)
    x = Conv2D(2 * width, 2, **conv_args, )(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm)(x)  # no activation

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='residual_concat_64x_33l_concat_' + name_suffix)
    return model, 64


def residual_32x_31l_concat(num_classes, name_suffix=''):
    """
    March 10+

    No dilation
    1x1Conv in common trunk

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
    width = 64

    def residual_block(x):
        y = Cropping2D(cropping=((1, 1), (1, 1)), )(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, **conv_args, )(x)

        x = Concatenate()([x, y])
        x = Conv2D(width, 1, **conv_args, )(x)
        return x

    x = Input(shape=(None, None, 3))
    input_layer = x

    x = Conv2D(width, 3, **conv_args, )(x)

    for i in range(14):
        x = residual_block(x)

    x = BatchNormalization()(x)
    x = Conv2D(width, 2, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm, )(x)  # no activation

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='residual_concat_32x_31l_' + name_suffix)
    return model, 32


def residual_32x_17l_concat(num_classes, name_suffix=''):
    """
    March 10+

    Dilation 2
    1x1 conv in common branch

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
    width = 64

    def residual_block(x):
        y = Cropping2D(cropping=((2, 2), (2, 2)), )(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, **conv_args, dilation_rate=2)(x)

        x = Concatenate()([x, y])
        x = Conv2D(width, 1, **conv_args)(x)
        return x

    x = Input(shape=(None, None, 3))  # None, None
    input_layer = x

    x = Conv2D(width, 3, **conv_args, )(x)

    for i in range(7):
        x = residual_block(x)

    x = BatchNormalization()(x)
    x = Conv2D(width, 2, **conv_args, )(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm)(x)  # no activation

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='residual_concat_32x_31l_' + name_suffix)
    return model, 32


def residual_64x(num_classes, name_suffix='', skip_branch_conv=True):
    """
    March 9

    Dilated 2

    1x1Conv in skip branch
    Adding branches

    :param skip_branch_conv:
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
    width = 64

    def residual_block(x, pooling=False):
        # skip-branch
        y = Cropping2D(cropping=((2, 2), (2, 2)), )(x)
        if skip_branch_conv:
            y = Conv2D(width, 1, **conv_args)(y)

        # main branch
        x = BatchNormalization()(x)
        x = Conv2D(width, 3, **conv_args, dilation_rate=2, )(x)

        if pooling:  # 4x in total
            x = MaxPool2D(pool_size=2, strides=1, padding='same')(x)

        x = add([x, y])
        return x

    x = Input(shape=(None, None, 3))  # None, None
    input_layer = x

    # note: No BatchNorm
    x = Conv2D(width, 3, activation='relu', padding='same',
               kernel_initializer=he_norm, )(x)

    for i in range(1, 16):
        x = residual_block(x, pooling=(i % 4 == 1))

    # note: No BatchNorm
    x = Conv2D(128, 4, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm)(x)  # todo run without the relu here
    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x, name='residual_64x' + name_suffix)
    return model, 64


def dilated_1(num_classes, name_suffix=''):
    """
    February 22
    :param num_classes:
    :param name_suffix:
    :return:
    """
    non_cropping_conv = True

    he_norm = tf.keras.initializers.he_normal()
    conv_args = {
        'activation': 'relu',
        'padding': 'valid',
        'kernel_initializer': he_norm
    }
    width = 128

    x = Input(shape=(None, None, 3))  # None, None
    input_layer = x

    # Note: No BatchNorm
    x = Conv2D(width, 3, activation='relu', padding='same',
               kernel_initializer=he_norm, )(x)

    for i in range(1, 6):
        y = Cropping2D(cropping=(i, i), name=f'crop{i}')(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, **conv_args, dilation_rate=i, )(x)

        if non_cropping_conv:
            x = BatchNormalization()(x)
            x = Conv2D(width, 3, activation='relu', padding='same',
                       kernel_initializer=he_norm, )(x)

        x = MaxPool2D(pool_size=2, strides=1, padding='same')(x)
        x = add([x, y])

    # Note: No BatchNorm
    x = Conv2D(width, 2, **conv_args, )(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, **conv_args)(x)
    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x, name='dilated' + name_suffix)
    return model, 32


def fcn_residual_32x_18l(num_classes, name_suffix=''):
    """
    February 15

    Adding residual branches

    :param num_classes:
    :param name_suffix:
    :return:
    """
    he_norm = tf.keras.initializers.he_normal()
    conv_args = {'activation': 'relu',
                 'padding': 'valid',
                 'kernel_initializer': he_norm}

    coef = 3
    width = 64

    input_layer = Input(shape=(None, None, 3))
    x = BatchNormalization()(input_layer)
    x = Conv2D(width, 3, **conv_args, )(x)  # Makes width wide enough for addition inside skip module

    for i in range(7):
        # width = 1 << (i // 4 + coef)  # ceil(log2(i / 2))  # todo try widening again
        y = Cropping2D(cropping=((2, 2), (2, 2)), )(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, **conv_args, )(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, **conv_args, )(x)

        # if i % 2 == 0:
        #     x = AvgPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

        # y = Conv2D(width, 1, **conv_args)(y)  # 1x1
        x = add([x, y])

    x = BatchNormalization()(x)
    x = Conv2D(16 * 1 << coef, 2, **conv_args)(x)  # fit-once

    x = BatchNormalization()(x)
    x = Conv2D(16 * 1 << coef, 1, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm, activation=None)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x, name='residual_32x_64w_18l_' + name_suffix)
    return model, 32


def fcn_sequential(num_classes, name_suffix=''):
    """
    January 25

    :param num_classes:
    :param name_suffix:
    :return:
    """
    conv_args = {'activation': 'relu',
                 'padding': 'valid',
                 'kernel_initializer': tf.keras.initializers.he_normal()}

    model = tf.keras.Sequential(name="fcn_16layers_" + name_suffix)

    model.add(tf.keras.Input(shape=(None, None, 3)))
    model.add(experimental.preprocessing.Rescaling(1. / 255))

    coef = 5
    for i in range(1, 16):
        width = 1 << (i // 4 + coef)  # ceil(log2(i / 2))
        # width ranges from 32 to 512

        model.add(BatchNormalization())
        model.add(Conv2D(width, 3, **conv_args, ))

        if i % 4 == 3:
            model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(BatchNormalization())
    model.add(Conv2D(16 * 1 << coef, 2, **conv_args))  # fit-once

    model.add(BatchNormalization())
    model.add(Conv2D(16 * 1 << coef, 1, **conv_args))

    model.add(BatchNormalization())
    model.add(Conv2D(num_classes, 1, **conv_args))

    model.add(Softmax())
    return model, 32


def fcn_maxpool_div(num_classes, weight_init_idx=0):
    """
    December/January

    Worked well
    output res 23 x 31

    Solved problems with accuracy not improving.
    Solved free points for uniform predictions.
    (util.accu)

    """
    weight_init = [tf.keras.initializers.he_uniform(),
                   tf.keras.initializers.he_normal(),
                   tf.keras.initializers.glorot_uniform()]
    print('Weight init:[{}] = {}'.format(weight_init_idx, weight_init[weight_init_idx].distribution))

    channels_base = 64
    width = np.array([1, 2, 2, 4, 4])  # multiplier for channels_base
    layers = len(width)
    name = 'FCN_layers{}_channels{}_init{}'.format(layers, width * channels_base, weight_init_idx)

    model = tf.keras.Sequential(name=safestr(name))
    model.add(tf.keras.Input(shape=(None, None, 3)))
    model.add(experimental.preprocessing.Rescaling(1. / 255))

    for i in range(layers):
        model.add(BatchNormalization(name='bn'))
        model.add(Conv2D(width[i] * channels_base,
                         3,
                         activation='relu',
                         padding='same',
                         kernel_initializer=weight_init[weight_init_idx],
                         ))
        model.add(MaxPooling2D())

    # [1 x 1 x C]

    model.add(Conv2D(128, 1, activation='relu'))
    model.add(Conv2D(num_classes, 1, activation='relu'))
    model.add(Softmax())

    return model, 32


def conv_failed_attempt(num_classes):
    """
    December
    does not train at all

    """
    c = 32
    model = tf.keras.Sequential([
        Input(shape=(None, None, 3)),
        experimental.preprocessing.Rescaling(1. / 255),
        Conv2D(c, 3, activation='relu'),
        Conv2D(c, 3, activation='relu'),
        MaxPooling2D(),

        Conv2D(2 * c, 3, activation='relu'),
        Conv2D(2 * c, 3, activation='relu'),
        Conv2D(2 * c, 3, activation='relu'),
        MaxPooling2D(),

        Conv2D(4 * c, 3, activation='relu'),
        Conv2D(4 * c, 3, activation='relu'),
        MaxPooling2D(),

        Conv2D(8 * c, 3, activation='relu'),
        MaxPooling2D(),

        # [1 x 1] here

        # Conv2D(128, 1, activation='relu'),
        Conv2D(num_classes, 1, activation='relu'),  # [W x H x many] -> [W x H x C]
        Softmax(),
        Flatten()
    ], name='sequential_9Conv_{}channels'.format(c))
    return model, 32


def conv_6layers(num_classes, first_conv=32):
    """
    November
    More layers and channels did not bring better accuracy.
    Is there a bottleneck in the model somewhere?

    """
    model = tf.keras.models.Sequential(
        [
            experimental.preprocessing.Rescaling(1. / 255),

            Conv2D(first_conv, 3, activation='relu', padding='same'),
            Conv2D(first_conv, 3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(first_conv * 2, 3, activation='relu', padding='same'),
            Conv2D(first_conv * 2, 3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(first_conv * 4, 3, activation='relu', padding='same'),
            Conv2D(first_conv * 4, 3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(128, activation='relu', kernel_initializer=init),
            # Dropout(0.2),
            Dense(num_classes, activation='softmax'),
        ], name='sequential_6conv_{}channels_doubling'.format(first_conv)
    )
    return model, 64


def conv_init(num_classes, first_conv=32):
    """
    November
    Trying out He initialization has not shown improvements so far.
    """
    he = tf.keras.initializers.he_normal()
    model = tf.keras.models.Sequential(
        [
            experimental.preprocessing.Rescaling(1. / 255),

            Conv2D(first_conv, 3, activation='relu', padding='same', kernel_initializer=he),
            Conv2D(first_conv, 3, activation='relu', padding='same', kernel_initializer=he),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(first_conv * 2, 3, activation='relu', padding='same', kernel_initializer=he),
            Conv2D(first_conv * 2, 3, activation='relu', padding='same', kernel_initializer=he),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(first_conv * 4, 3, activation='relu', padding='same', kernel_initializer=he),
            Conv2D(first_conv * 4, 3, activation='relu', padding='same', kernel_initializer=he),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(128, activation='relu', kernel_initializer=he),
            Dense(num_classes, activation='softmax'),
        ], name='sequential_6conv_{}channels_doubling'.format(first_conv)
    )
    return model, 64


def conv_tutorial(num_classes, input_dim=(64, 64, 3)):
    """
    October/November
    So far using mostly this.
    Best accuracy=0.9809 after 44 epochs (on a very small dataset)

    """
    c = 32  # 32
    model = tf.keras.Sequential([
        Input(shape=input_dim),
        experimental.preprocessing.Rescaling(1. / 255),
        Conv2D(c, 3, activation='relu', padding='same'),
        MaxPooling2D(),

        Conv2D(c, 3, activation='relu', padding='same'),
        MaxPooling2D(),

        Conv2D(c, 3, activation='relu', padding='same'),
        MaxPooling2D(),

        Flatten(),
        # Dropout(0.2),

        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ], name='sequential_3conv_{}channels'.format(c))
    return model, 64
