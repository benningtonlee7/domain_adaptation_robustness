"""
Base models for:
    ADDA: Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
    WDGRL: Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
    REVGRAD: Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
    ALDA: Adversarial-Learned Loss for Domain Adaptation Chen et al. (2020)
"""

from torch import nn
from torch.autograd import Function


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = False
        self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(5, 5)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(5, 5)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=(5, 5)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout2d(0.5),
                )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = False
        self.classifier = nn.Sequential(
                nn.Linear(32 * 10 * 10, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 10),
                )

    def forward(self, x):
        out = self.classifier(x)
        return out


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = False
        self.layers = nn.Sequential(
            nn.Linear(32 * 10 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, alpha=None):
        if alpha:
            x = ReverseLayerF.apply(x, alpha)
        out = self.layers(x)
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
