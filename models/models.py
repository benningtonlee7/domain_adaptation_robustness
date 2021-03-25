"""
Base models for:
    ADDA: Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
    WDGRL: Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
    REVGRAD: Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
"""
from torch import nn
from torch.autograd import Function

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = False
        self.feature_extractor = nn.Sequential(
                # First conv layer: in_dim 1x28x28, out_dim 20x12x12
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                # Second conv layer: in_dim 20x12x12, out_dim 50x4x4
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU()
                )

    def forward(self, x):
        features = self.feature_extractor(x)
        return features

class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = False
        self.layers = nn.Sequential(
                nn.Linear(50 * 4 * 4, 500),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(500, 10)
                )

    def forward(self, x):
        out = self.layers(x.view(-1, 50 * 4 * 4))
        return out

"""
Discriminator model
"""
class Discriminator(nn.Module):

    def __init__(self, in_dims, h_dims, out_dims):
        super().__init__()
        self.pretrained = False
        self.layers = nn.Sequential(
            nn.Linear(in_dims, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, out_dims),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.layers(x.view(-1, 50 * 4 * 4))
        return out


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None