# stdlib
import os
import sys
from model_util import tensorboard_hparams_init, tf_init

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime

# external libs
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorboard.plugins.hparams import api as hp

# local files
from models import *
import model_build
from datasets import get_dataset
from eval_images import eval_full_predictions_all
from eval_samples import evaluate_model
from display import show_layer_activations
from logs import log_model_info
from src_util.util import safestr, DuplicateStream, get_checkpoint_path
import config


def run(model_builder, hparams):
    """Perform a single training run"""

    try:
        dataset_dir = '{}x_{:03d}s_{}bg' \
            .format(config.dataset_dim, int(100 * config.scale), config.dataset_size)
        dataset_dir = os.path.join(config.datasets_dir, dataset_dir)
        print('Loading dataset from:', dataset_dir)
        config.checkpoint_path = get_checkpoint_path()

        time = safestr(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        """ Load dataset """
        use_weights = hparams['class_weights'] if 'class_weights' in hparams else None
        val_ds, _, _ = get_dataset(dataset_dir + '_val', batch_size=config.batch_size)
        train_ds, config.class_names, class_weights = \
            get_dataset(dataset_dir, weights=use_weights, batch_size=config.batch_size)

        """ Create/Load a model """
        if config.train:
            base_model, model, aug_model = model_build.build_new_model(model_builder, hparams, name_suffix=time)
            callbacks = model_build.get_callbacks()

        else:
            load_model_name = '64x_d1-3-5-7-9-11-1-1_2021-05-10-05-53-28_full'  # /data/datasets/128x_025s_1000bg

            base_model, model, aug_model = model_build.load_model(load_model_name)
            callbacks = model_build.get_callbacks()
            config.epochs_trained = 123

        """ Model outputs dir """
        config.output_location = os.path.join('outputs', model.name + '_reloaded' * (not config.train))
        if not os.path.isdir(config.output_location):
            os.makedirs(config.output_location, exist_ok=True)

        """ Copy stdout to file """
        stdout_orig = sys.stdout
        out_stream = open(os.path.join(config.output_location, 'stdout.txt'), 'a')
        sys.stdout = DuplicateStream(sys.stdout, out_stream)

        print({h: hparams[h] for h in hparams})

        log_model_info(model, config.output_location, config.models_saved_dir)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.get_logger().setLevel('ERROR')  # suppress warnings about early-stopping and model-checkpoints

        config.epochs_trained = 0

        if config.train:
            try:
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    validation_freq=5,
                    epochs=(config.epochs + config.epochs_trained),
                    initial_epoch=config.epochs_trained,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=2  # one line per epoch
                )
            except KeyboardInterrupt:
                print('Training stopped preemptively')
            config.epochs_trained += config.epochs
            try:
                if config.epochs_trained > 5:
                    model.load_weights(config.checkpoint_path)  # checkpoint
            except NotFoundError:
                # might not be present if trained for <K epochs
                pass

            base_model.save_weights(os.path.join('models_saved', model.name))

        """Evaluate model"""

        # config.show = True
        # config.output_location = None  # do-not-save flag
        val_accu = evaluate_model(model, val_ds, val=True, output_location=config.output_location,
                                  show=config.show, misclassified=False)

        evaluate_model(model, train_ds, val=False, output_location=config.output_location, show=config.show)

        if val_accu < 95.0:  # %
            print('Val accu too low:', val_accu, 'skipping heatmaps')
            return

        """Full image prediction"""
        pix_mse_val, dist_mse_val, keypoint_count_mae_val, crate_count_mae_val, crate_count_failrate_val = \
            eval_full_predictions_all(base_model, val=True, output_location=config.output_location, show=config.show)
        pix_mse_train, dist_mse_train, keypoint_count_mae_train, crate_count_mae_train, crate_count_failrate_train = \
            eval_full_predictions_all(base_model, val=False, output_location=config.output_location, show=False)

        val_metrics = model.evaluate(val_ds, verbose=0)  # 5 metrics, as per model_ops.compile_model
        pr_value_val = val_metrics[4]

        print('mse: {}'.format(dist_mse_train))
        print('mse_val: {}'.format(dist_mse_val))
        print('pr_value_val: {}'.format(pr_value_val))

        if config.train:
            with tf.summary.create_file_writer(config.run_logs_dir + '/hparams').as_default():
                hp.hparams(hparams, trial_id=config.model_name)

                tf.summary.scalar('pix_mse_train', pix_mse_train, step=config.epochs_trained)
                tf.summary.scalar('point_dist_mse_train', dist_mse_train, step=config.epochs_trained)
                tf.summary.scalar('point_count_mae_train', keypoint_count_mae_train, step=config.epochs_trained)
                tf.summary.scalar('crate_count_mae_train', crate_count_mae_train, step=config.epochs_trained)
                tf.summary.scalar('crate_count_failrate_train', crate_count_failrate_train, step=config.epochs_trained)

                tf.summary.scalar('pix_mse_val', pix_mse_val, step=config.epochs_trained)
                tf.summary.scalar('point_dist_mse_val', dist_mse_val, step=config.epochs_trained)
                tf.summary.scalar('point_count_mae_val', keypoint_count_mae_val, step=config.epochs_trained)
                tf.summary.scalar('crate_count_mae_val', crate_count_mae_val, step=config.epochs_trained)
                tf.summary.scalar('crate_count_failrate_val', crate_count_failrate_val, step=config.epochs_trained)

                tf.summary.scalar('pr_value_val', pr_value_val, step=config.epochs_trained)

        """Per layer activations"""
        show_layer_activations(base_model, aug_model, val_ds, show=False,
                               output_location=config.output_location)

        # restore original stdout
        sys.stdout = stdout_orig

    except Exception as ex:
        print(type(ex), ex, '\n\n', file=sys.stderr)


if __name__ == '__main__':
    tf_init()

    config.dataset_size = 1000
    config.train = True
    # train dim decided by model
    config.dataset_dim = 128
    config.augment = 3
    config.show = False
    config.scale = 0.25
    config.center_crop_fraction = 0.5
    config.epochs = 50

    # model params
    hp_base_width = hp.HParam('base_width', hp.Discrete([8, 16, 32]))
    # non-model params
    hp_aug_level = hp.HParam('aug_level', hp.Discrete([0, 1, 2, 3]))
    hp_class_weights = hp.HParam('ds_bg_samples', hp.Discrete(['none', 'inverse_frequency', 'effective_number']))
    hp_crop_fraction = hp.HParam('crop_fraction', hp.Discrete([0.5, 1.0]))
    hp_ds_bg_samples = hp.HParam('ds_bg_samples', hp.Discrete([200, 700, 1000]))
    hp_scale = hp.HParam('scale', hp.Discrete([0.25, 0.5]))
    hparams = [
        hp_aug_level,
        hp_base_width,
        hp_class_weights,
        hp_crop_fraction,
        hp_ds_bg_samples,
        hp_scale,
    ]
    tensorboard_hparams_init(hparams)

    m = parameterized(recipe_51x_odd)
    hparams = {
        'base_width': 32,

        'aug_level': config.augment,
        'ds_bg_samples': config.dataset_size,
        'scale': config.scale,
        'crop_fraction': config.center_crop_fraction,
    }
    run(m, hparams)

    pass
