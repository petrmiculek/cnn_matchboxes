# stdlib
import os

# external
import tensorflow as tf

from tensorflow.keras.callbacks import \
    TensorBoard, ReduceLROnPlateau, \
    LearningRateScheduler, ModelCheckpoint

# local
import util
import models
import config


def compile_model(model):
    """ Custom loss and metrics """
    scce_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    accu = util.Accu(name='accu')  # ~= SparseCategoricalAccuracy
    prec = util.Precision(name='prec')
    recall = util.Recall(name='recall')
    pr_value = util.AUC(name='pr_value', curve='PR')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=scce_loss,
        metrics=[accu,
                 prec,
                 recall,
                 pr_value,
                 ])


def load_model(model_name, load_weights=True):
    model_config_path = os.path.join('outputs', model_name, 'model_config.json')
    weights_path = os.path.join('models_saved', model_name)

    with open(model_config_path, mode='r') as config_file:
        config_json = config_file.read()

    model = tf.keras.models.model_from_json(config_json,
                                            custom_objects={'RandomColorDistortion': util.RandomColorDistortion})

    compile_model(model)

    data_augmentation = model.layers[0]
    base_model = model.layers[1]
    if load_weights:
        base_model.load_weights(weights_path)

    return base_model, model, data_augmentation


def get_callbacks():
    """ TensorBoard loggging """
    config.run_logs_dir = os.path.join('logs', f'bg{config.dataset_size}', config.model_name)
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

    lr_logging = util.LearningRateLogger()

    mse_logging = util.MSELogger(freq=5)

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
    aug_model = models.augmentation(aug_level=hparams['aug_level'], crop_to=config.train_dim, ds_dim=config.dataset_dim)
    config.model_name = base_model.name + '_full'
    model = tf.keras.Sequential([aug_model, base_model], name=config.model_name)
    compile_model(model)

    return base_model, model, aug_model
