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
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])


def conv_tutorial_tweaked(num_classes):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),

            tf.keras.layers.Conv2D(4, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(8, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(16, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes),
            tf.keras.layers.Softmax()

        ]
    )
