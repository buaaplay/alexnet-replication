# utils/dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloaders(data_dir, batch_size=128, num_workers=4, cifar100=False):
    print(f"get_dataloaders called with cifar100={cifar100}")

    if cifar100:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), 
                                 (0.2673, 0.2564, 0.2762)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), 
                                 (0.2673, 0.2564, 0.2762)),
        ])

        train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
