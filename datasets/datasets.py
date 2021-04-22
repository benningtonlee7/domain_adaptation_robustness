import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from utils.utils import gray2rgb
import params
from scipy.io import loadmat


def get_mnist(train, download=True, drop_last=True, get_pseudo=False):
    """Get MNIST dataset loader."""
    # Image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((32, 32)),
                                      transforms.Lambda(gray2rgb),
                                      transforms.Normalize(
                                          mean=(params.dataset_mean,),
                                          std=(params.dataset_std,))])
    # Dataset and data loader
    if get_pseudo:
        path = 'data/mnist_train_pseudo.mat' if train else 'data/mnist_eval_pseudo.mat'
        mnist_dataset = CustomDataset(path, transforms.ToTensor())
    else:
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


def get_usps(train, download=True, drop_last=True, get_pseudo=False):
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
    if get_pseudo:
        path = 'data/usps_train_pseudo.mat' if train else 'data/usps_eval_pseudo.mat'
        usps_dataset = CustomDataset(path, transforms.ToTensor())
    else:
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


def get_svhn(split='Train', download=True, drop_last=True, get_pseudo=False):
    """Get SVHN dataset loader."""
    # Image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((32, 32)),
                                      transforms.Normalize(
                                          mean=(params.dataset_mean,),
                                          std=(params.dataset_std,))])

    # Dataset and data loader
    if get_pseudo:
        path = 'data/svhn_train_pseudo.mat' if split == 'Train' else 'data/svhn_eval_pseudo.mat'
        svhn_dataset = CustomDataset(path, transforms.ToTensor())
    else:
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


class CustomDataset(Dataset):
    def __init__(self, mat_file, transform=None):
        """
        Args:
            mat_file (string): Path to the mat file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = loadmat(mat_file)
        self.images = data['images']
        self.labels = torch.from_numpy(data['labels']).squeeze()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.images[idx]
        labels = self.labels[idx]
        if self.transform:
            imgs = self.transform(imgs)

        return imgs, labels
