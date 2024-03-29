# external
import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, BatchNormalization, Softmax, \
    Input, MaxPool2D

# local
import model_util
import src_util.util

# common model building blocks
he_norm = tf.keras.initializers.he_normal()
conv_args = {
    'activation': 'relu',
    'padding': 'valid',
    'kernel_initializer': he_norm
}


def dilated_64x_odd(num_classes=8, hparams=None, name_suffix=''):
    """
    March 15

    Dilation rate growth accounts for receptive field growth from pooling

    :param num_classes:
    :param hparams:
    :param name_suffix:
    :return:
    """
    if hparams is None:
        hparams = {}
    base_width = hparams['base_width'] if 'base_width' in hparams else 16

    dilations = [1, 3, 5, 7, 9, 11, 1, 1]
    widths = [2, 4, 4, 8, 8, 16, 16, 16]
    kernels = [3, 3, 3, 3, 3, 2, 3, 1]

    x = Input(shape=(None, None, 3))  # 64, 64
    input_layer = x

    for k, d, w in zip(kernels, dilations, widths):
        width = base_width * w
        x = Conv2D(width, k, **conv_args, dilation_rate=d)(x)
        x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm, activation=None)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='64x_odd_' + name_suffix)
    return model, 64


def dilated_32x_odd(num_classes=8, hparams=None, name_suffix=''):
    """
    March 15

    Dilation rate growth accounts for receptive field growth from pooling

    :param num_classes:
    :param hparams:
    :param name_suffix:
    :return:
    """
    if hparams is None:
        hparams = {}

    base_width = hparams['base_width'] if 'base_width' in hparams else 16

    tail_downscale = hparams['tail_downscale'] if 'tail_downscale' in hparams else 1

    x = Input(shape=(None, None, 3))  # 32, 32
    input_layer = x

    # non-cropping conv
    x = Conv2D(base_width, 3, kernel_initializer=he_norm, activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    dilations = [3, 5, 7]
    widths = [2, 4, 8 // tail_downscale]

    for d, w in zip(dilations, widths):
        x = Conv2D(base_width * w, 3, **conv_args, dilation_rate=d)(x)

        x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)

    x = Conv2D(8 * base_width // tail_downscale, 2, **conv_args)(x)  # fit-once

    x = BatchNormalization()(x)
    x = Conv2D(8 * base_width // tail_downscale, 1, **conv_args)(x)

    x = BatchNormalization()(x)
    x = Conv2D(num_classes, 1, kernel_initializer=he_norm, activation=None)(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x,
                           name='32x_odd_' + name_suffix)
    return model, 32


def parameterized(recipe=None):
    """
    :param recipe:
    :return: model object, model input dimension
    """
    if recipe is None:
        raise Exception('No model recipe')

    kernels, dilations, widths = recipe()
    input_dim = recipe_input_dim(kernels, dilations)
    param_string = '-'.join((str(d) for d in dilations))

    def model_parameterized(num_classes=8, hparams=None, name_suffix=''):
        if hparams is None:
            hparams = {}

        base_width = hparams['base_width'] if 'base_width' in hparams else 16

        x = Input(shape=(None, None, 3))  # input_dim, input_dim  # <- use for model.summary() to see layer sizes
        input_layer = x

        if 'non_cropping_conv' in hparams and hparams['non_cropping_conv']:
            x = Conv2D(base_width, 3, kernel_initializer=he_norm, activation='relu', padding='same')(x)
            x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)

        for k, d, w in zip(kernels, dilations, widths):
            x = Conv2D(w * base_width, k, **conv_args, dilation_rate=d)(x)

            x = MaxPool2D((2, 2), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)

        x = Conv2D(num_classes, 1, kernel_initializer=he_norm, activation=None)(x)
        x = Softmax()(x)

        name = '{}x_d{}_{}'.format(input_dim, param_string, name_suffix)

        model = tf.keras.Model(inputs=input_layer, outputs=x,
                               name=name)
        return model, input_dim

    return model_parameterized


def recipe_input_dim(kernels, dilations, *args):
    dim = 1

    for k, d in zip(kernels, dilations):
        dim -= general.conv_dim_calc(w=0, k=k, d=d)

    if not (dim & (dim - 1) == 0 and dim > 1):
        print("W: model recipe does not result in a power of 2 input size")

    return dim


def recipe_64x_odd():
    """

    Dilation rate increases ideally with pooling rate
    :return:
    """
    dilations = [1, 3, 5, 7, 9, 11] + [1, 1]
    widths = [2, 4, 4, 8, 8, 16] + [16, 16]
    kernels = [3, 3, 3, 3, 3, 2] + [3, 1]

    return kernels, dilations, widths


# def recipe_64x_exp2():
#     """
#
#     Dilation rate increases ideally with pooling rate
#     :return:
#     """
#     dilations = [1, 2, 4, 8, 16, ] + [1, 1]
#     widths = [2, 4, 4, 8, 8, 16] + [16, 16]
#     kernels = [3, 3, 3, 3, 3, 2] + [3, 1]
#
#     return kernels, dilations, widths


def recipe_64x_basic():
    """

    dilation_rate = 1 => no dilation
    17 conv layers
    :return:
    """
    kernels = [3] * 15 + [2, 1]
    dilations = [1] * 15 + [1, 1]
    widths = [2] * 5 + [4] * 5 + [8] * 5 + [8, 8]

    return kernels, dilations, widths


def recipe_32x_flat5():
    kernels = [3, 3, 3, 2, 1]
    dilations = [5, 5, 5, 1, 1]
    widths = [2, 4, 8, 8, 8]

    return kernels, dilations, widths


def recipe_32x_flat2():
    """

    dilation_rate = 1 => no dilation
    17 conv layers
    :return:
    """
    kernels = [3] * 7 + [3, 2, 1]
    dilations = [2] * 7 + [1, 1, 1]
    widths = [2] * 2 + [4] * 4 + [8] * 4

    return kernels, dilations, widths


def recipe_32x_d1to5():
    kernels = [3, 3, 3, 3, 3] + [2, 1]
    dilations = [1, 2, 3, 4, 5] + [1, 1]
    widths = [2, 2, 4, 4, 8] + [8, 8]

    return kernels, dilations, widths


def recipe_32x_exp2():
    kernels = [3, 3, 3, 3] + [2, 1]
    dilations = [1, 2, 4, 8] + [1, 1]
    widths = [2, 4, 4, 8] + [8, 8]

    return kernels, dilations, widths


def recipe_32x_d7531():
    kernels = [3, 3, 3, 2, 1]
    dilations = [7, 5, 3, 1, 1]
    widths = [2, 4, 4, 8, 8]

    return kernels, dilations, widths


def recipe_32x_d1357_start2():
    kernels = [2, 3, 3, 3, 1]
    dilations = [1, 3, 5, 7, 1]
    widths = [2, 4, 4, 8, 8]

    return kernels, dilations, widths


def recipe_32x_odd():
    """

    First conv+pooling block does not crop
    (not included in lists below)

    When using, add to hparams:
    hparams['non_cropping_conv'] = True

    'non_cropping_conv': True,


    :return:
    """
    kernels = [3, 3, 3, 2, 1]
    dilations = [3, 5, 7, 1, 1]
    widths = [4, 4, 8, 8, 8]

    return kernels, dilations, widths


def recipe_sandbox():
    dilations = [1, 2, 3, 4, 5]
    kernels = [3] * len(dilations)
    dim = 32
    print(dim)
    for k, d in zip(kernels, dilations):
        dim += general.conv_dim_calc(w=0, k=k, d=d)
        print(dim)


def metarecipe_odd(growth=None, layers=4):
    dilation_rate = 1
    k = 3
    kernels = []
    dilations = []
    widths = []
    for i in range(layers):
        kernels.append(k)
        dilations.append(dilation_rate)
        if growth == 'odd':
            dilation_rate += 2
        elif growth == 'exp':
            dilation_rate *= 2

        widths.append(8 * 2 ** (i // 2))

    def recipe_from_dilation():
        return kernels, dilations, widths

    return recipe_from_dilation


def recipe_51x_odd():
    dilations = [1, 3, 5, 7, 9, 1]
    kernels = [3, 3, 3, 3, 3, 1]
    widths = [2, 2, 4, 4, 8, 8]

    return kernels, dilations, widths


def recipe_73x_odd():
    dilations = [1, 3, 5, 7, 9, 11, 1]
    kernels = [3, 3, 3, 3, 3, 3, 1]
    widths = [2, 2, 4, 4, 8, 8, 8]

    return kernels, dilations, widths


def recipe_99x_odd():
    dilations = [1, 3, 5, 7, 9, 11, 13, 1]
    kernels = [3, 3, 3, 3, 3, 3, 3, 1]
    widths = [2, 2, 4, 4, 8, 8, 8, 8]

    return kernels, dilations, widths

