import tensorflow as tf


def conv_tutorial(num_classes):
    """
    So far using mostly this.
    Best accuracy=0.9809 after 44 epochs

    """
    c = 32  # 32
    return tf.keras.Sequential([
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
    he = tf.keras.initializers.HeNormal()
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),

            tf.keras.layers.Conv2D(first_conv, 3, activation='relu', padding='same', kernel_initializer=init),
            tf.keras.layers.Conv2D(first_conv, 3, activation='relu', padding='same', kernel_initializer=init),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(first_conv * 2, 3, activation='relu', padding='same', kernel_initializer=init),
            tf.keras.layers.Conv2D(first_conv * 2, 3, activation='relu', padding='same', kernel_initializer=init),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(first_conv * 4, 3, activation='relu', padding='same', kernel_initializer=init),
            tf.keras.layers.Conv2D(first_conv * 4, 3, activation='relu', padding='same', kernel_initializer=init),
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
