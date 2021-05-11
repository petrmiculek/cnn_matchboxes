# stdlib
import os

# external
import sys
import warnings

import tensorflow as tf

from tensorflow.keras.callbacks import \
    TensorBoard, ReduceLROnPlateau, \
    LearningRateScheduler, ModelCheckpoint
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import \
    RandomFlip, RandomRotation, RandomZoom, \
    CenterCrop, RandomTranslation

# local
import model_util
import config


def compile_model(model):
    """ Custom loss and metrics """
    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = model_util.Accu(name='accu')  # ~= SparseCategoricalAccuracy
    prec = model_util.Precision(name='prec')
    recall = model_util.Recall(name='recall')
    pr_value = model_util.AUC(name='pr_value', curve='PR')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=scce_loss,
        metrics=[accu,
                 prec,
                 recall,
                 pr_value,
                 ])


def load_model(model_name, models_saved_dir=None, load_weights=True):
    if models_saved_dir is None:
        models_saved_dir = config.models_saved_dir

    model_config_path = os.path.join(models_saved_dir, f'{model_name}.json')
    weights_path = os.path.join(models_saved_dir, model_name)

    with open(model_config_path, mode='r') as config_file:
        config_json = config_file.read()

    custom_objects = {'RandomColorDistortion': model_util.RandomColorDistortion}
    model = tf.keras.models.model_from_json(config_json, custom_objects=custom_objects)

    compile_model(model)

    data_augmentation = model.layers[0]
    base_model = model.layers[1]
    if load_weights:
        base_model.load_weights(weights_path)

    return base_model, model, data_augmentation


def get_callbacks():
    """ TensorBoard loggging """
    config.run_logs_dir = os.path.join(config.logs_dir, 'bg{}'.format(config.dataset_size), config.model_name)
    os.makedirs(config.run_logs_dir, exist_ok=True)
    file_writer = tf.summary.create_file_writer(config.run_logs_dir + "/metrics")
    file_writer.set_as_default()

    """ Callbacks """
    # lr_sched = LearningRateScheduler(util.lr_scheduler)  # unused
    reduce_lr = ReduceLROnPlateau(monitor='pr_value', mode='max', factor=0.5, min_delta=1e-2,
                                  patience=5, min_lr=5e-6, verbose=tf.compat.v1.logging.ERROR)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_pr_value', patience=15, min_delta=1e-3, mode='max',
                                                      verbose=tf.compat.v1.logging.ERROR)  # avoid not available warning
    # ^ No improvement on training data for 5 epochs -> Reduce LR
    # No improvement on validation data for 15 epochs -> Halt

    lr_logging = model_util.LearningRateLogger()

    mse_logging = model_util.EvalLogger(freq=5)

    tensorboard_callback = TensorBoard(config.run_logs_dir, write_graph=False,
                                       histogram_freq=1, profile_batch=0)  # profiling needs sudo
    # tf.debugging.experimental.enable_dump_debug_info(config.run_logs_dir,
    #                                                  tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    # tf.debugging.set_log_device_placement(True)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=config.checkpoint_path,
        save_weights_only=True,
        monitor='val_accu',
        mode='max',
        save_best_only=True)

    callbacks = [
        lr_logging,
        reduce_lr,
        tensorboard_callback,
        model_checkpoint_callback,
        mse_logging,
        early_stopping,
    ]

    return callbacks


def build_new_model(model_factory, hparams, name_suffix=''):
    base_model, config.train_dim = model_factory(len(config.class_names),
                                                 hparams=hparams, name_suffix=name_suffix)
    aug_model = augmentation(aug_level=hparams['aug_level'], crop_to=config.train_dim, ds_dim=config.dataset_dim)
    config.model_name = base_model.name + '_full'
    model = tf.keras.Sequential([aug_model, base_model], name=config.model_name)
    compile_model(model)

    return base_model, model, aug_model


def augmentation(aug_level=2, crop_to=64, ds_dim=64):
    """

    RandomZoom
    A positive value means zooming out, while a negative value means zooming in.

    :param aug_level: augmentation level 0, 1, 2, 3; default=2=values mentioned in text
    :param crop_to: output dimension == native base_model input dimension
    :param ds_dim: input dimension == dataset dimension
    :return: Keras sequential model
    """
    if ds_dim < crop_to:
        raise ValueError('E: Augmentation: Dataset dim ({}) smaller than target dim ({})'.format(ds_dim, crop_to))

    if aug_level > 0 and ds_dim == crop_to:
        warnings.warn('W: Augmentation: Model input and output dimensions are the same.' +
                      '\nAugmentations may create artifacts around image edges (e.g. black corners)')

    aug_model = tf.keras.Sequential(name='augmentation')
    aug_model.add(Input(shape=(ds_dim, ds_dim, 3)))

    e = sys.float_info.epsilon

    # parameters for augmentation-levels 0..3
    brightness = [0.0, 0.1, 0.3, 0.5]
    contrast = [(1.0, 1.0 + e), (0.75, 1.25), (0.5, 1.5), (0.35, 2.0)]
    hue = [0, 0.1, 0.2, 0.4]
    saturation = [(1.0, 1.0 + e), (0.75, 1.25), (0.5, 1.5), (0.1, 1.5)]
    zoom = [(0.0, 0.0 + e), (-0.1, +0.1), (-0.25, 0.25), (-0.5, +0.5)]
    # aug-level independent parameters
    rotation = 1 / 16  # =rot22.5°
    translation = [0, 1 / 64, 2 / 64, 4 / 64]  # e.g. 4 pixels for a 64x model

    if aug_level > 0:
        aug_model.add(RandomFlip("horizontal"))
        aug_model.add(RandomTranslation(translation[aug_level], translation[aug_level]))
        aug_model.add(RandomRotation(rotation))
        aug_model.add(model_util.RandomColorDistortion(brightness_delta=brightness[aug_level],
                                                       contrast_range=contrast[aug_level],
                                                       hue_delta=hue[aug_level],
                                                       saturation_range=saturation[aug_level]))
        aug_model.add(RandomZoom(zoom[aug_level], fill_mode='constant'))

    if ds_dim != crop_to:
        aug_model.add(CenterCrop(crop_to, crop_to))

    if aug_level == 0 and crop_to == ds_dim:
        # no other layers, model cannot be empty
        # base Layer class == identity layer
        aug_model.add(tf.keras.layers.Layer(name='identity'))

    return aug_model
