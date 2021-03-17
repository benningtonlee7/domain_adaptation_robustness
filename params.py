"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = [dataset_mean_value, dataset_mean_value, dataset_mean_value]
dataset_std = [dataset_std_value, dataset_std_value, dataset_std_value]
batch_size = 50
image_size = 64

# params for source dataset
src_encoder_path = "snapshots/ADDA-source-encoder-final.pt"
clf_path = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_encoder_path = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_in_dims = 500
d_h_dims = 500
d_out_dims = 2
disc_path = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
num_epochs_pre = 100
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 1000
log_step = 100
save_step = 100
manual_seed = None
lr_max = 0.01
early_stop = True

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

# params for PGD attack
upper_limit = 1
lower_limit = 0
clamps = [lower_limit, upper_limit]
attack_iters = 10
pgd_alpha = 2
restarts = 1
norm = 'l_inf'
epsilon = 8

# params for wdgrl
wd_clf = 0.1

