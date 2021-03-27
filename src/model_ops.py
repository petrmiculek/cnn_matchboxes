import os
import tensorflow as tf

from tensorflow.keras.callbacks import \
    TensorBoard, ReduceLROnPlateau, \
    LearningRateScheduler, ModelCheckpoint

import util
import models
from show_results import heatmaps_all


def compile_model(model):
    """ Custom loss and metrics """
    # scce_loss = util.Scce(name='scce_loss')  # test tensor shapes
    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = util.Accu(name='accu')  # ~= SparseCategoricalAccuracy
    prec = util.Precision(name='prec')
    recall = util.Recall(name='recall')
    pr_curve = util.AUC(name='pr_curve', curve='PR')
    # f1 = util.F1(num_classes=8)  # todo maybe 2 classes
    # , average='micro'
    # todo parametrize ^ based on model's last layer

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=scce_loss,
        metrics=[accu,
                 prec,
                 recall,
                 pr_curve,
                 # f1,
                 ])


def load_model(config, weights=None):
    with open(config, mode='r') as config_file:
        config_json = config_file.read()

    model = tf.keras.models.model_from_json(config_json,
                                            custom_objects={'RandomColorDistortion': util.RandomColorDistortion})

    # dim = model.layers[0].layers[0].input_shape
    # model.build(input_shape=(dim, dim, 3))

    compile_model(model)

    data_augmentation = model.layers[0]
    base_model = model.layers[1]
    if weights:
        base_model.load_weights(weights)

    callbacks = get_callbacks(model)

    return base_model, model, data_augmentation, callbacks


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr


class MAELogger(tf.keras.callbacks.Callback):
    def __init__(self, class_names, freq=1):
        super().__init__()
        self._supports_tf_logs = True
        self.class_names = class_names
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and (epoch - 1) % self.freq == self.freq - 1:
            base_model = self.model.layers[1]
            heatmaps_all(base_model, self.class_names, val=True, output_location=None, show=False,
                         epochs_trained=epoch)


def get_callbacks(model, checkpoint_path='/tmp/checkpoint', bg_samples='_unknown'):
    """ TensorBoard loggging """
    logs_dir = os.path.join('logs', f'bg{bg_samples}', model.name)
    os.makedirs(logs_dir, exist_ok=True)
    file_writer = tf.summary.create_file_writer(logs_dir + "/metrics")
    file_writer.set_as_default()

    """ Callbacks """
    # lr_sched = LearningRateScheduler(util.lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='accu', factor=0.5, min_delta=1e-2,
                                  patience=10, min_lr=5e-6)

    lr_logging = LearningRateLogger()

    # todo ugly
    class_names = ['background',
                   'corner-bottom',
                   'corner-top',
                   'edge-bottom',
                   'edge-side',
                   'edge-top',
                   'intersection-side',
                   'intersection-top']

    mae_logging = MAELogger(class_names, freq=15)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accu', patience=15, min_delta=1e-3, mode='max')

    # ^ No improvement on training data for 10 epochs -> Reduce LR,
    # No improvement on validation data for 15 epochs -> Halt

    tensorboard_callback = TensorBoard(logs_dir, histogram_freq=1, profile_batch='100,1100')
    # tf.debugging.experimental.enable_dump_debug_info(logs_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    # tf.debbuging.set_log_device_placement(True)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accu',
        mode='max',
        save_best_only=True)

    callbacks = [
        lr_logging,
        reduce_lr,
        tensorboard_callback,
        model_checkpoint_callback,
        mae_logging,
        early_stopping,
    ]

    return callbacks


def build_new_model(model_factory, model_kwargs, num_classes=8, augment=True, name_suffix='',
                    ds_dim=64, checkpoint_path='/tmp/checkpoint', bg_samples='_unknown'):
    base_model, training_dim = model_factory(num_classes, **model_kwargs, name_suffix=name_suffix)
    data_augmentation = models.augmentation(aug=augment, crop_to=training_dim, ds_dim=ds_dim)
    model = tf.keras.Sequential([data_augmentation, base_model], name=base_model.name + '_full')
    compile_model(model)
    callbacks = get_callbacks(model, checkpoint_path, bg_samples)

    return base_model, model, data_augmentation, callbacks
