import os
import random
import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
import params
from scipy.io import savemat

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


def make_variable(tensor):
    """Convert Tensor to Variable."""
    tensor = tensor.to(DEVICE)
    return tensor


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def gray2rgb(tensor):
    return tensor.repeat(3, 1, 1)


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


def normalize(X, std=params.dataset_std, mean=params.dataset_mean):
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
    alpha = alpha.expand(h_s.size()).type_as(h_s)
    try:
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates = torch.cat((interpolates, h_s, h_t), dim=0).requires_grad_()

        preds = critic(interpolates)
        gradients = grad(preds, interpolates,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gp = ((gradient_norm - 1)**2).mean()
    except:
        gp = 0
    return gp


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def create_matrix(n):
    """
    :param n: matrix size (class num)
    :return a matrix with torch.tensor type:
    for example n=3:
    1     -1/2  -1/2
    -1/2    1   -1/2
    -1/2  -1/2    1
    """
    a = np.zeros((n,n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i==j:
                a[i,j]=1
            else:
                a[i,j]=-1/(n-1)
    return torch.from_numpy(a).cuda()


def alda_loss(preds_critic, labels_src, preds_src, preds_tgt, weight_type=1, threshold=0.9):

    preds_concat = torch.cat((preds_src, preds_tgt), dim=0)
    softmax_out = nn.Softmax(dim=1)(preds_concat)
    ad_out = torch.sigmoid(preds_critic)
    batch_size = preds_critic.size(0) // 2
    class_num = preds_critic.size(1)

    labels_source_mask = torch.zeros(batch_size, class_num).to(DEVICE).scatter_(1, labels_src.unsqueeze(1), 1)
    probs_source = softmax_out[:batch_size].detach()
    probs_target = softmax_out[batch_size:].detach()
    maxpred, argpred = torch.max(probs_source, dim=1)
    preds_source_mask = torch.zeros(batch_size, class_num).to(DEVICE).scatter_(1, argpred.unsqueeze(1), 1)
    maxpred, argpred = torch.max(probs_target, dim=1)
    preds_target_mask = torch.zeros(batch_size, class_num).to(DEVICE).scatter_(1, argpred.unsqueeze(1), 1)

    # Filter out those low confidence samples
    target_mask = (maxpred > threshold)
    preds_target_mask = torch.where(target_mask.unsqueeze(1), preds_target_mask, torch.zeros(1).to(DEVICE))

    # Construct the confusion matrix from ad_out. See the paper for more details.
    confusion_matrix = create_matrix(class_num)
    ant_eye = (1 - torch.eye(class_num)).cuda().unsqueeze(0)
    # (2*batch_size, class_num, class_num)
    confusion_matrix = ant_eye / (class_num - 1) + torch.mul(confusion_matrix.unsqueeze(0), ad_out.unsqueeze(1))
    preds_mask = torch.cat([preds_source_mask, preds_target_mask], dim=0)  # labels_source_mask
    loss_pred = torch.mul(confusion_matrix, preds_mask.unsqueeze(1)).sum(dim=2)

    # Different correction targets for different domains
    loss_target = (1 - preds_target_mask) / (class_num - 1)
    loss_target = torch.cat([labels_source_mask, loss_target], dim=0)
    if not ((loss_pred >= 0).all() and (loss_pred <= 1).all()):
        raise AssertionError
    mask = torch.cat([(maxpred >= 0), target_mask], dim=0)
    adv_loss = nn.BCELoss(reduction='none')(loss_pred, loss_target)[mask]
    adv_loss = torch.sum(adv_loss) / mask.float().sum()

    # Reg_loss
    reg_loss = nn.CrossEntropyLoss()(preds_critic[:batch_size], labels_src)

    # corrected target loss function
    target_probs = 1.0 * softmax_out[batch_size:]
    correct_target = torch.mul(confusion_matrix.detach()[batch_size:], preds_target_mask.unsqueeze(1)).sum(dim=2)
    correct_loss = -torch.mul(target_probs, correct_target)
    correct_loss = torch.mean(correct_loss[target_mask])
    return adv_loss, reg_loss, correct_loss


def pseudo_label(encoder, classifier, dataset_name, data_loader):

    encoder.eval()
    classifier.eval()
    x, y = [], []
    for step, (images, _) in enumerate(data_loader):
        images = make_variable(images)
        preds = classifier(encoder(images))
        _, preds = torch.max(preds, 1)
        for i in range(images.size(0)):
            x.append(images[i, :, :, :].permute(1, 2, 0).cpu().numpy())
            y.append([preds[i].cpu().numpy()])

    out = {"images": x, "labels": y}
    savemat(os.path.join("data", dataset_name + ".mat"), out)


def update_lr(optimizer, lr):

    for g in optimizer.param_groups:
        g.update(lr=lr)


def save_model(model, model_root=params.model_root, filename='model'):
    
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    torch.save(model.state_dict(),
               os.path.join(model_root, filename))
    print("Save model to: {}".format(os.path.join(model_root, filename)))
