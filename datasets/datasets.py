import torch
from torchvision import datasets, transforms
import params


def get_mnist(train, download=True):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                      transforms.Normalize(
                                          mean=params.dataset_mean_value,
                                          std=params.dataset_std_value)])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=download)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return mnist_data_loader

def get_usps(train, download=True):
    """Get USPS dataset loader."""
    # Image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(28),
                                      transforms.Normalize(
                                          mean=params.dataset_mean_value,
                                          std=params.dataset_std_value)])

    # dataset and data loader
    usps_dataset = datasets.USPS(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=download)

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return usps_data_loader

def get_svhn(split='Train', download=True):
    """Get SVHN dataset loader."""
    # Image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(28),
                                      transforms.Grayscale(1),
                                      transforms.Normalize(
                                       mean=params.dataset_mean_value,
                                       std=params.dataset_std_value)])

    # Dataset and data loader
    svhn_dataset = datasets.SVHN(root=params.data_root,
                                   split=split,
                                   transform=pre_process,
                                   download=download)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return svhn_data_loader
