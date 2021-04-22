# Params for dataset and data loader
data_root = "data"
dataset_mean = 0.5
dataset_std = 0.5
batch_size = 150
num_worker = 2

# Params for training network
num_epochs_pre = 20
log_step_pre = 50  # log every # steps
eval_step_pre = 5  # eval every # epoch
save_step_pre = 5  # save every # epoch
num_epochs = 100  # epochs for training
log_step = 50  # log every # steps
eval_step = 10  # Eval every # epoch
save_step = 10  # save every # epoch
manual_seed = None
lr_max = 0.01
early_stop = False

# Params for optimizing models
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0005

# Params for PGD attack
upper_limit = 1
lower_limit = 0
clamps = [lower_limit, upper_limit]
attack_iters = 10
pgd_alpha = 2
restarts = 1
norm = 'l_inf'
epsilon = 8

# Params for wdgrl
wd_clf = 0.1
wd_gp = 10
num_times_critic = 10
beta_ratio = 0

# Params for saving models
model_root = "snapshots"
model_trained = True

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
disc_wdgrl_path = "snapshots/WDGRL/WDGRL-critic-final.pt"
encoder_wdgrl_path = "snapshots/WDGRL/WDGRL-encoder-final.pt"
clf_wdgrl_path = "snapshots/WDGRL/WDGRL-classifier-final.pt"

disc_wdgrl_rb_path = "snapshots/WDGRL/WDGRL-critic-rb-final.pt"
encoder_wdgrl_rb_path = "snapshots/WDGRL/WDGRL-encoder-rb-final.pt"
clf_wdgrl_rb_path = "snapshots/WDGRL/WDGRL-classifier-rb-final.pt"

# DANN
dann_root = "snapshots/DANN"
tgt_encoder_dann_path = "snapshots/DANN/DANN-encoder-final.pt"
clf_dann_path = "snapshots/DANN/DANN-classifier-final.pt"
disc_dann_path = "snapshots/DANN/DANN-critic-final.pt"

tgt_encoder_dann_rb_path = "snapshots/DANN/DANN-encoder-rb-final.pt"
clf_dann_rb_path = "snapshots/DANN/DANN-classifier-rb-final.pt"
disc_dann_rb_path = "snapshots/DANN/DANN-critic-rb-final.pt"

# Standard adversarily trained model
tgt_encoder_rb_path = "snapshots/ADV-source-encoder-rb-final.pt"
clf_rb_path = "snapshots/ADV-source-classifier-rb-final.pt"

# Not adversarily trained
tgt_encoder_path = "snapshots/STANDARD-source-encoder-final.pt"
clf_path = "snapshots/STANDARD-source-classifier-final.pt"