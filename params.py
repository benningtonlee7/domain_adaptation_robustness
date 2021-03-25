# Params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = [dataset_mean_value, dataset_mean_value, dataset_mean_value]
dataset_std = [dataset_std_value, dataset_std_value, dataset_std_value]
batch_size = 128

# Params for models
model_root = "snapshots"
model_trained = True
d_in_dims = 500
d_h_dims = 500
d_out_dims = 2

# ADDA
adda_root = "snapshots/ADDA"
src_encoder_adda_path = "snapshots/ADDA/ADDA-source-encoder-final.pt"
clf_adda_path = "snapshots/ADDA/ADDA-source-classifier-final.pt"
tgt_encoder_adda_path = "snapshots/ADDA/ADDA-target-encoder-final.pt"
disc_adda_path = "snapshots/ADDA/ADDA-critic-final.pt"

src_encoder_adda_rb_path = "snapshots/ADDA/ADDA-source-encoder-rb-final.pt"
clf_adda_rb_path = "snapshots/ADDA/ADDA-source-classifier-rb-final.pt"
tgt_encoder_adda_rb_path = "snapshots/ADDA/ADDA-target-encoder-rb-final.pt"
disc_adda_rb_path = "snapshots/ADDA/ADDA-critic-rb-final.pt"

# WDGRL
wdgrl_root = "snapshots/WDGRL"
disc_wdgrl_path = "snapshots/WDGRL/WDGRL-critic-rb-final.pt"
encoder_wdgrl_path = "snapshots/WDGRL/WDGRL-encoder-rb-final.pt"
clf_wdgrl_path = "snapshots/WDGRL/WDGRL-classifier-rb-final.pt"

disc_wdgrl_rb_path = "snapshots/WDGRL/WDGRL-critic-rb-final.pt"
encoder_wdgrl_rb_path = "snapshots/WDGRL/WDGRL-encoder-rb-final.pt"
clf_wdgrl_rb_path = "snapshots/WDGRL/WDGRL-classifier-rb-final.pt"

# REVGARD
revgard_root = "snapshots/REVGARD"
tgt_encoder_revgrad_path = "snapshots/REVGRAD/REVGRAD-encoder-final.pt"
clf_revgrad_path = "snapshots/REVGRAD/REVGRAD-classifier-final.pt"

tgt_encoder_revgrad_rb_path = "snapshots/REVGRAD/REVGRAD-encoder-rb-final.pt"
clf_revgrad_rb_path = "snapshots/REVGRAD/REVGRAD-classifier-rb-final.pt"

# DANN
dann_root = "snapshots/DANN"
tgt_encoder_dann_path = "snapshots/REVGRAD/REVGRAD-encoder-final.pt"
clf_dann_path = "snapshots/REVGRAD/REVGRAD-classifier-final.pt"

tgt_encoder_dann_rb_path = "snapshots/REVGRAD/REVGRAD-encoder-rb-final.pt"
clf_dann_rb_path = "snapshots/REVGRAD/REVGRAD-classifier-rb-final.pt"

# Params for training network
num_epochs_pre = 50
log_step_pre = 10  # log every # steps
eval_step_pre = 25  # eval every # epoch
save_step_pre = 10  # save every # epoch
num_epochs = 200  # epochs for training
log_step = 20  # log every # steps
eval_step = 50  # Eval every # epoch
save_step = 50  # save every # epoch
manual_seed = None
lr_max = 0.01
early_stop = False

# Params for optimizing models
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

# Params for PGD attack
upper_limit = 1
lower_limit = 0
clamps = [lower_limit, upper_limit]
attack_iters = 10
pgd_alpha = 2
restarts = 2
norm = 'l_inf'
epsilon = 8

# Params for wdgrl
wd_clf = 0.1
num_epochs_wdgrl_pre = 10

# Params for dann
momentum = 0.9
lr = 0.01
weight_decay = 1e-6