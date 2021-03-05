import tensorflow as tf
from tensorflow.keras.layers import \
    add, Conv2D, BatchNormalization, Softmax, Input, MaxPool2D, Cropping2D
from tensorflow.keras.layers.experimental.preprocessing import \
    CenterCrop, RandomFlip, RandomRotation
import numpy as np
from math import log2, ceil
import util
from src_util.general import safestr
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler


def get_model(model_factory, num_classes=8, name_suffix='', logs_dir='logs/unknown'):
    base_model = model_factory(num_classes, name_suffix=name_suffix)
    base_model.summary()

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        RandomFlip("horizontal"),
        RandomRotation(1/8, fill_mode='reflect'),
        util.RandomColorDistortion(),
        CenterCrop(32, 32),
    ], name='aug_only')

    model = tf.keras.Sequential([data_augmentation, base_model], name=base_model.name + '_full')
    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = util.Accu(name='accu')  # ~= SparseCategoricalAccuracy

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=scce_loss,
        metrics=[accu,
                 # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name='t'),
                 # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False, name='f'),
                 ])

    # lr_sched = LearningRateScheduler(util.lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='accu', factor=0.2,
                                  patience=5, min_lr=1e-7)

    tensorboard_callback = TensorBoard(logs_dir, histogram_freq=1, profile_batch='300,400')
    callbacks = [
        tensorboard_callback,
        tf.keras.callbacks.EarlyStopping(monitor='accu',
                                         patience=10),
        # lr_sched,
        reduce_lr,
    ]

    return base_model, model, data_augmentation, callbacks


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

    x = Conv2D(width, 3, activation='relu', padding='same', kernel_initializer=he_norm)(x)

    for i in range(1, 6):
        y = Cropping2D(cropping=(i, i))(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, **conv_args, dilation_rate=i)(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, activation='relu', padding='same', kernel_initializer=he_norm)(x)

        x = MaxPool2D(pool_size=2, strides=1, padding='same')(x)
        x = add([x, y])

    x = Conv2D(width, 2, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, **conv_args)(x)
    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x, name='dilated' + name_suffix)
    return model


def fcn_residual_1(num_classes, name_suffix=''):
    """
    February 15

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
    # width = 8

    input_layer = Input(shape=(None, None, 3))
    x = BatchNormalization()(input_layer)
    x = Conv2D(width, 3, **conv_args)(x)  # Makes width wide enough for addition inside skip module

    for i in range(1, 8):
        # width = 1 << (i // 4 + coef)  # ceil(log2(i / 2))  # todo try widening again
        y = Cropping2D(cropping=((2, 2), (2, 2)))(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, **conv_args)(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, 3, kernel_initializer=he_norm)(x)  # todo try again with **conv_args

        # if i % 2 == 0:  # todo try again
        #     x = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

        x = add([x, y])
        # x = Conv2D(width, 1, **conv_args)(x)  # 1x1  # todo try again

    x = BatchNormalization()(x)
    x = Conv2D(16 * 1 << coef, 2, **conv_args)(x)  # fit-once

    x = BatchNormalization()(x)
    x = Conv2D(16 * 1 << coef, 1, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, **conv_args)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x, name='residual_' + name_suffix)
    return model


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
        model.add(Conv2D(width, 3, **conv_args))

        if i % 4 == 3:
            model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(BatchNormalization())
    model.add(Conv2D(16 * 1 << coef, 2, **conv_args))  # fit-once

    model.add(BatchNormalization())
    model.add(Conv2D(16 * 1 << coef, 1, **conv_args))

    model.add(BatchNormalization())
    model.add(Conv2D(num_classes, 1, **conv_args))

    model.add(Softmax())
    return model


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
        model.add(BatchNormalization())
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

    return model


def conv_failed_attempt(num_classes, input_dim=(64, 64, 3)):
    """
    December
    does not train at all

    """
    c = 32
    return tf.keras.Sequential([
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


def conv_6layers(num_classes, first_conv=32):
    """
    November
    More layers and channels did not bring better accuracy.
    Is there a bottleneck in the model somewhere?

    """
    return tf.keras.models.Sequential(
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

        ]
        , name='sequential_6conv_{}channels_doubling'.format(first_conv)
    )


def conv_init(num_classes, first_conv=32):
    """
    November
    Trying out He initialization has not shown improvements so far.
    """
    he = tf.keras.initializers.he_normal()
    return tf.keras.models.Sequential(
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

        ]
        , name='sequential_6conv_{}channels_doubling'.format(first_conv)
    )


def conv_tutorial(num_classes, input_dim=(64, 64, 3)):
    """
    October/November
    So far using mostly this.
    Best accuracy=0.9809 after 44 epochs (on a very small dataset)

    """
    c = 32  # 32
    return tf.keras.Sequential([
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


def basic_conv_ablation(num_classes, input_dim=(32, 32, 3)):
    """
    March 4
    """
    c = 32  # 32
    he_norm = tf.keras.initializers.he_normal()
    conv_args = {
        'activation': 'relu',
        'padding': 'same',
        'kernel_initializer': he_norm
    }

    return tf.keras.Sequential([
        Input(shape=input_dim),
        experimental.preprocessing.Rescaling(1. / 255),
        Conv2D(c, 3, activation='relu', padding='same'),
        MaxPooling2D(),

        BatchNormalization(),
        Conv2D(c, 3, activation='relu', padding='same'),
        MaxPooling2D(),

        Conv2D(8, 1, activation='relu', padding='same'),
        MaxPooling2D(),
        Softmax()
    ], name='sequential_3conv_{}channels'.format(c))
