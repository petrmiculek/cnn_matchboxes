import tensorflow as tf
from tensorflow.keras.layers import *  # add, Conv2D, BatchNormalization, Softmax, experimental, Input, MaxPool2D
import numpy as np
from math import log2, ceil
import util


def fcn_dilated(num_classes, name_suffix=''):
    model = tf.keras.Sequential(name='dilated')
    model.add(Input(shape=(None, None, 3)))
    dilation = 1
    for i in range(8):
        width = 16
        model.add(Conv2D(width, activation='relu', padding='valid', dilation_rate=dilation))


def fully_fully_conv(num_classes, name_suffix='', weight_init_idx=0):
    init = tf.keras.initializers.he_normal()
    model = tf.keras.Sequential(name="fcn_16layers_" + name_suffix)

    model.add(tf.keras.Input(shape=(None, None, 3)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255))
    # width = [8,
    #          16,
    #          32, 32,
    #          64, 64, 64, 64,
    #          128, 128, 128, 128, 128, 128, 128, ]
    coef = 5
    for i in range(1, 16):
        width = 1 << (i // 4 + coef)  # ceil(log2(i / 2))
        # width = 64
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(width,
                                         3,
                                         activation='relu',
                                         padding='valid',
                                         kernel_initializer=init,
                                         ))
        if i % 4 == 3:
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(16 * 1 << coef, 2, activation='relu', padding='valid',
                                     kernel_initializer=init))  # fit-once

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(16 * 1 << coef, 1, activation='relu', padding='valid', kernel_initializer=init))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(num_classes, 1, activation='relu', padding='valid', kernel_initializer=init))

    model.add(tf.keras.layers.Softmax())
    return model


def fully_conv_tff(num_classes, name_suffix=''):
    """
    rewriting fully_fully_conv into TF.Functional
    not used yet

    :param num_classes:
    :param name_suffix:
    :return:
    """

    init = tf.keras.initializers.he_normal()
    # name="fcn_16layers_" + name_suffix
    coef = 4

    # x = Input(shape=(None, None, 3))
    input_layer = Input(shape=(None, None, 3))
    x = experimental.preprocessing.Rescaling(1. / 255)(input_layer)

    # x = Conv2D(width,
    #            3,
    #            activation='relu',
    #            padding='valid',
    #            kernel_initializer=init)(x)
    width = 128

    x = BatchNormalization()(x)
    x = Conv2D(width,
               3,
               activation='relu',
               padding='valid',
               kernel_initializer=init)(x)

    for i in range(1, 8):
        # width = 1 << (i // 4 + coef)  # ceil(log2(i / 2))
        x = BatchNormalization()(x)
        y = Cropping2D(cropping=((2, 2), (2, 2)))(x)

        x = Conv2D(width,
                   3,
                   activation='relu',
                   padding='valid',
                   kernel_initializer=init)(x)

        x = Conv2D(width,
                   3,
                   activation='relu',
                   padding='valid',
                   kernel_initializer=init)(x)

        if i % 2 == 0:
            x = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

        x = tf.keras.layers.add([x, y])
        # x = Conv2D(width, 1, activation='relu', padding='valid', kernel_initializer=init)(x)  # 1x1

    x = BatchNormalization()(x)
    x = Conv2D(16 * 1 << coef, 2, activation='relu', padding='valid', kernel_initializer=init)(x)  # fit-once

    x = BatchNormalization()(x)
    x = Conv2D(16 * 1 << coef, 1, activation='relu', padding='valid', kernel_initializer=init)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, activation='relu', padding='valid', kernel_initializer=init)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x, name='tff_w_{}_l_{}'.format(width, 25) + name_suffix)
    return model


def fully_conv_maxpool_div(num_classes, weight_init_idx=0):
    """
    Worked well
    output res 23 x 31

    Solved problems with accuracy not improving.
    Solved free points for uniform predictions.
    (accu)

    """
    weight_init = [tf.keras.initializers.he_uniform(),
                   tf.keras.initializers.he_normal(),
                   tf.keras.initializers.glorot_uniform()]
    print('Weight init:[{}] = {}'.format(weight_init_idx, weight_init[weight_init_idx].distribution))

    channels_base = 64
    width = np.array([1, 2, 2, 4, 4])
    layers = len(width)
    name = 'FCN_layers{}_channels{}_init{}'.format(layers, width * channels_base, weight_init_idx)

    model = tf.keras.Sequential(name=util.safestr(name))
    model.add(tf.keras.Input(shape=(None, None, 3)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255))

    for i in range(layers):
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(width[i] * channels_base,
                                         3,
                                         activation='relu',
                                         padding='same',
                                         kernel_initializer=weight_init[weight_init_idx],
                                         ))
        model.add(tf.keras.layers.MaxPooling2D())

    # [1 x 1 x C]

    model.add(tf.keras.layers.Conv2D(128, 1, activation='relu'))
    model.add(tf.keras.layers.Conv2D(num_classes, 1, activation='relu'))
    model.add(tf.keras.layers.Softmax())

    return model


def fully_conv_failed_attempt(num_classes, input_dim=(64, 64, 3)):
    """
    does not train at all

    """
    c = 32
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, None, 3)),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(c, 3, activation='relu'),
        tf.keras.layers.Conv2D(c, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(2 * c, 3, activation='relu'),
        tf.keras.layers.Conv2D(2 * c, 3, activation='relu'),
        tf.keras.layers.Conv2D(2 * c, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(4 * c, 3, activation='relu'),
        tf.keras.layers.Conv2D(4 * c, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(8 * c, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        # [1 x 1] here

        # tf.keras.layers.Conv2D(128, 1, activation='relu'),
        tf.keras.layers.Conv2D(num_classes, 1, activation='relu'),  # [W x H x many] -> [W x H x C]
        tf.keras.layers.Softmax(),
        tf.keras.layers.Flatten()
    ], name='sequential_3fullyConv_{}channels'.format(c))


def conv_tutorial_deeper(num_classes, first_conv=32):
    """
    More layers and channels did not bring better accuracy.
    Is there a bottleneck in the model somewhere?

    """
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),

            tf.keras.layers.Conv2D(first_conv, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(first_conv, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(first_conv * 2, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(first_conv * 2, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(first_conv * 4, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(first_conv * 4, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer=init),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax'),

        ]
        , name='sequential_4conv_{}channels_doubling'.format(first_conv)
    )


def conv_tutorial_init(num_classes, first_conv=32):
    """
    Trying out He initialization has not shown improvements so far.
    """
    he = tf.keras.initializers.he_normal()
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),

            tf.keras.layers.Conv2D(first_conv, 3, activation='relu', padding='same', kernel_initializer=he),
            tf.keras.layers.Conv2D(first_conv, 3, activation='relu', padding='same', kernel_initializer=he),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(first_conv * 2, 3, activation='relu', padding='same', kernel_initializer=he),
            tf.keras.layers.Conv2D(first_conv * 2, 3, activation='relu', padding='same', kernel_initializer=he),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(first_conv * 4, 3, activation='relu', padding='same', kernel_initializer=he),
            tf.keras.layers.Conv2D(first_conv * 4, 3, activation='relu', padding='same', kernel_initializer=he),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer=he),
            tf.keras.layers.Dense(num_classes, activation='softmax'),

        ]
        , name='sequential_4conv_{}channels_doubling'.format(first_conv)
    )


def conv_tutorial(num_classes, input_dim=(64, 64, 3)):
    """
    So far using mostly this.
    Best accuracy=0.9809 after 44 epochs

    """
    c = 32  # 32
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_dim),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(c, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(c, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(c, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='sequential_3conv_{}channels'.format(c))
