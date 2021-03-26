import params
import torch.nn.functional as F
from models.models import ReverseLayerF
from utils.utils import clamp, normalize
import torch

def attack_pgd(encoder, classifier, X, y, dann=None, epsilon=params.epsilon, alpha=params.pgd_alpha,
               attack_iters=params.attack_iters, restarts=params.restarts,
               norm=params.norm, early_stop=params.early_stop, clamps=params.clamps):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon /= 255.0
    alpha /= 255.0
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.uniform_(-0.5,0.5).renorm(p=2, dim=1, maxnorm=epsilon)
        else:
            raise ValueError
        delta = clamp(delta, clamps[0]-X, clamps[1]-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            feats = encoder((normalize(X + delta)))
            if dann:
                feats = ReverseLayerF.apply(feats.view(-1, 50 * 4 * 4), dann)
            output = classifier(feats)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, clamps[0] - x, clamps[1] - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(classifier(encoder(normalize(X+delta))), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    return max_delta
