import os
import tensorflow as tf

from tensorflow.keras.callbacks import \
    TensorBoard, ReduceLROnPlateau, \
    LearningRateScheduler, ModelCheckpoint

import util
import models
import run_config


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
        optimizer=tf.keras.optimizers.Adam(learning_rate=run_config.learning_rate),
        loss=scce_loss,
        metrics=[accu,
                 prec,
                 recall,
                 pr_curve,
                 # f1,
                 ])


def load_model(model_config_path, weights_path=None):
    with open(model_config_path, mode='r') as config_file:
        config_json = config_file.read()

    model = tf.keras.models.model_from_json(config_json,
                                            custom_objects={'RandomColorDistortion': util.RandomColorDistortion})

    compile_model(model)

    data_augmentation = model.layers[0]
    base_model = model.layers[1]
    if weights_path:
        base_model.load_weights(weights_path)

    return base_model, model, data_augmentation


def get_callbacks():
    """ TensorBoard loggging """
    run_config.run_logs_dir = os.path.join('logs', f'bg{run_config.dataset_size}', run_config.model_name)
    os.makedirs(run_config.run_logs_dir, exist_ok=True)
    # file_writer = tf.summary.create_file_writer(run_config.run_logs_dir + "/metrics")
    # file_writer.set_as_default()

    """ Callbacks """
    # lr_sched = LearningRateScheduler(util.lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='accu', mode='max', factor=0.5, min_delta=1e-2,
                                  patience=10, min_lr=5e-6, verbose=tf.compat.v1.logging.ERROR)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accu', patience=15, min_delta=1e-3, mode='max',
                                                      verbose=tf.compat.v1.logging.ERROR)  # avoid not available warning
    # ^ No improvement on training data for 10 epochs -> Reduce LR,
    # No improvement on validation data for 15 epochs -> Halt

    lr_logging = util.LearningRateLogger()

    mse_logging = util.MSELogger(freq=5)

    tensorboard_callback = TensorBoard(run_config.run_logs_dir, histogram_freq=1, )  # profile_batch='1100, 3100' # needs sudo
    # tf.debugging.experimental.enable_dump_debug_info(run_config.run_logs_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    # tf.debugging.set_log_device_placement(True)

    # hparam_callback = hp.KerasCallback(logdir, hparams)  # todo check out

    model_checkpoint_callback = ModelCheckpoint(
        filepath=run_config.checkpoint_path,
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
    base_model, run_config.train_dim = model_factory(len(run_config.class_names),
                                                     hparams=hparams, name_suffix=name_suffix)
    aug_model = models.augmentation(aug=run_config.augment, crop_to=run_config.train_dim, ds_dim=run_config.dataset_dim)
    run_config.model_name = base_model.name + '_full'
    model = tf.keras.Sequential([aug_model, base_model], name=run_config.model_name)
    compile_model(model)

    return base_model, model, aug_model
