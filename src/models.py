import tensorflow as tf
import numpy as np
from math import log2


def fully_conv(num_classes, weight_init_idx=0):
    """
    Currently used model

    Accuracy does not seem to improve during training.

    """
    weight_init = [tf.keras.initializers.he_uniform(),
                   tf.keras.initializers.he_normal(),
                   tf.keras.initializers.glorot_uniform()]
    print('Weight init:[{}] = {}'.format(weight_init_idx, weight_init[weight_init_idx].distribution))

    channels_base = 64
    width = np.array([1, 2, 2, 4, 4, 4])
    layers = 6
    name = 'FCN_layers{}_channels{}_init{}'.format(layers, width * channels_base, weight_init_idx)
    name = name.replace(' ', '_').replace('[', '_').replace(']', '_')  # list elements contain spaces

    model = tf.keras.Sequential(name=name)
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


def fully_conv_tutorial(num_classes, input_dim=(64, 64, 3)):
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


def residual1(num_classes, first_conv=32):
    """
    WIP ResNet-like model
    """

    def res_block(channels=first_conv):
        pass

    return tf.keras.models.Functional()
    pass


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
