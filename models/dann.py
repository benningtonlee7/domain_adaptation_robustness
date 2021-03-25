import torch.nn as nn
from torch.autograd import Function
"""
Implements DANN: Unsupervised Domain Adaptation by Backpropagation
"""

class DANNEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = False

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

    def forward(self, x):

        x = x.expand(x.data.shape[0], 1, 28, 28)
        features = self.feature_extractor(x)
        return features


class DANNClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = False

        self.class_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = x.view(-1, 50 * 4 * 4)
        output = self.class_classifier(x)
        return output


class DANNDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained = False

        self.layers = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x, alpha):
        reverse_feature = ReverseLayerF.apply(x, alpha)
        domain_out = self.layers(reverse_feature)
        return domain_out


