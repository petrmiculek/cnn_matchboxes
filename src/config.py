# Global config module

### Before-run configuration

## Training configuration
import os

## Dataset

# Before loading
scale = 0.5
dataset_size = 1000
dataset_dim = 128  # load resolution
train_dim = 64  # resolution for training

# After loading
class_names = None

# Training flags
train = True
learning_rate = 1e-3
batch_size = 128
augment = 2
epochs = 20
class_weights = None

# Model versions
# base_model = None
# aug_model = None
# model = None
model_name = 'None'

# Model run statistics
epochs_trained = 0

# Evaluation
show = False
center_crop_fraction = 1.0

# directories
checkpoint_path = None
output_location = None
run_logs_dir = None

logs_root = 'logs'
datasets_root = '/data/datasets/'
