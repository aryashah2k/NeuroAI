import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataloaders(data_dir: str = ".data",
                           batch_size: int = 128,
                           num_workers: int = 2,
                           drop_last: bool = False):
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])

    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader
