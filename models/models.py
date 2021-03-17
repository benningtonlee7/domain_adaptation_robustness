"""
Base models for:
    ADDA: Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
    WDGRL: Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
    REVGRAD: Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
"""
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """LeNet feature extractor model"""

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
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, x):
        out = self.feature_extractor(x)
        features = self.fc1(out.view(-1, 50 * 4 * 4))
        return features

class Classifier(nn.Module):
    """LeNet Classifier model"""
    
    def __init__(self):
        super().__init__()
        self.pretrained = False
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = F.relu(x)
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
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
            nn.LogSoftmax()
        )

    def forward(self, X):
        out = self.layers(X)
        return out
