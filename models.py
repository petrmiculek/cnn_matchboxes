import tensorflow as tf


def conv_tutorial(num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='sequential_3conv_32channels')


def conv_tutorial_tweaked(num_classes, first_conv=32):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),

            tf.keras.layers.Conv2D(first_conv, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # == default pool_size

            tf.keras.layers.Conv2D(first_conv * 2, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(first_conv * 4, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(first_conv * 8, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax'),

        ]
        , name='sequential_4conv_{}channels_doubling'.format(first_conv)
    )
