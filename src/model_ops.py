import os
import tensorflow as tf

from tensorflow.keras.callbacks import \
    TensorBoard, ReduceLROnPlateau, \
    LearningRateScheduler, ModelCheckpoint

import util
import models


def compile_model(model):
    """ Custom loss and metrics """
    # scce_loss = util.Scce(name='scce_loss')  # test tensor shapes
    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = util.Accu(name='accu')  # ~= SparseCategoricalAccuracy
    prec = util.Precision(name='prec')
    recall = util.Recall(name='recall')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=scce_loss,
        metrics=[accu,
                 prec,
                 recall,
                 # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name='t'),
                 # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False, name='f'),
                 # tf.keras.metrics.AUC(name='auc'),
                 ])


def load_model(config, weights=None):
    with open(config, mode='r') as config_file:
        config_json = config_file.read()

    model = tf.keras.models.model_from_json(config_json)

    # dim = model.layers[0].layers[0].input_shape
    # model.build(input_shape=(dim, dim, 3))

    compile_model(model)

    data_augmentation = model.layers[0]
    base_model = model.layers[1]
    if weights:
        base_model.load_weights(weights)

    callbacks = get_callbacks(model)

    return base_model, model, data_augmentation, callbacks


def get_callbacks(model, checkpoint_path='/tmp/checkpoint', bg_samples='_unknown'):

    """ TensorBoard loggging """
    logs_dir = os.path.join('logs', f'bg{bg_samples}', model.name)
    os.makedirs(logs_dir, exist_ok=True)
    file_writer = tf.summary.create_file_writer(logs_dir + "/metrics")
    file_writer.set_as_default()

    """ Callbacks """
    # lr_sched = LearningRateScheduler(util.lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='accu', factor=0.5,
                                  patience=10, min_lr=5e-6)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accu', patience=15)

    # ^ No improvement on training data for 10 epochs -> Reduce LR,
    # No improvement on validation data for 15 epochs -> Halt

    tensorboard_callback = TensorBoard(logs_dir, histogram_freq=1, profile_batch='300,400')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accu',
        mode='max',
        save_best_only=True)

    callbacks = [
        tensorboard_callback,
        early_stopping,
        # lr_sched,
        reduce_lr,
        model_checkpoint_callback
    ]

    return callbacks


def build_new_model(model_factory, num_classes=8, augment=True, name_suffix='', checkpoint_path='/tmp/checkpoint', bg_samples=100):
    base_model, training_dim = model_factory(num_classes, name_suffix=name_suffix)
    data_augmentation = models.augmentation(aug=augment, crop_to=training_dim)
    model = tf.keras.Sequential([data_augmentation, base_model], name=base_model.name + '_full')
    compile_model(model)
    callbacks = get_callbacks(model, checkpoint_path, bg_samples)

    return base_model, model, data_augmentation, callbacks
