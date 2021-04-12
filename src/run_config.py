# Global config module

### Before-run configuration

## Training configuration
import os

train = None
augment = None
use_weights = None

## Dataset

# Before loading
scale = None
dataset_size = None
dataset_dim = None  # load resolution
train_dim = None  # resolution for training

# After loading
class_names = None

# Training flags


# Model versions
# base_model = None
# aug_model = None
# model = None
model_name = None

# Model run statistics
epochs_trained = None

# Evaluation
show = False
center_crop_fraction = 1.0
learning_rate = 1e-3

# directories
checkpoint_path = None
output_location = None
run_logs_dir = None

logs_hparams_dir = os.path.join('logs', 'hparams')  # todo unused
