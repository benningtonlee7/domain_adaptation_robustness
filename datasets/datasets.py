import torch
from torchvision import datasets, transforms
from utils.utils import gray2rgb
import params


def get_mnist(train, download=True, drop_last=True):
    """Get MNIST dataset loader."""
    # Image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((32, 32)),
                                      transforms.Lambda(gray2rgb),
                                      transforms.Normalize(
                                          mean=(params.dataset_mean,),
                                          std=(params.dataset_std,))])

    # Dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=download)
    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_worker,
        drop_last=drop_last,
        pin_memory=True)

    return mnist_data_loader

def get_usps(train, download=True, drop_last=True):
    """Get USPS dataset loader."""
    # Image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((32, 32)),
                                      transforms.Lambda(gray2rgb),
                                      transforms.Normalize(
                                          mean=(params.dataset_mean,),
                                          std=(params.dataset_std,)),
                                      ])

    # Dataset and data loader
    usps_dataset = datasets.USPS(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=download)

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_worker,
        drop_last=drop_last,
        pin_memory=True)

    return usps_data_loader

def get_svhn(split='Train', download=True, drop_last=True):
    """Get SVHN dataset loader."""
    # Image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((32, 32)),
                                      transforms.Normalize(
                                          mean=(params.dataset_mean,),
                                          std=(params.dataset_std,))])

    # Dataset and data loader
    svhn_dataset = datasets.SVHN(root=params.data_root,
                                   split=split,
                                   transform=pre_process,
                                   download=download)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_worker,
        drop_last=drop_last,
        pin_memory=True)

    return svhn_data_loader
