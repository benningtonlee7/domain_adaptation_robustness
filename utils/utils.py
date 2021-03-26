"""Utility functions for ADDA"""

import os
import random
import torch
from torch.autograd import Variable, grad
import params

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    tensor = tensor.to(DEVICE)
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

def lr_piecewise(t, num_epochs):
    if t / num_epochs < 0.5:
        return params.lr_max
    elif t / num_epochs < 0.75:
        return params.lr_max / 10.0
    else:
        return params.lr_max / 100.0


def lr_scheduler(p):
    lr_0 = 0.01
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    return lr


def clamp(X, lower_limit=params.lower_limit, upper_limit=params.upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def normalize(X, std=params.dataset_std_value, mean=params.dataset_mean_value):
    return (X - mean)/std


def model_init(model, restore=None):
    model.apply(init_weights)
    # Load state dict
    if restore and os.path.exists(restore):
        model.pretrained = True
        model.load_state_dict(torch.load(restore))
        print("Load model from: {}".format(os.path.abspath(restore)))
    
    return model.to(DEVICE)


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(DEVICE)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    grad_penalty = ((gradient_norm - 1) ** 2).mean()

    return grad_penalty

def update_lr(optimizer, lr):

    for g in optimizer.param_groups:
        g.update(lr=lr)


def save_model(model, model_root, filename):
    
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    torch.save(model.state_dict(),
               os.path.join(model_root, filename))
    print("Save model to: {}".format(os.path.join(model_root, filename)))
