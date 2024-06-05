import torch
from torch import nn
from torch.utils.data import dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root = 'Data',
    train = True,
    download=True,
    transform=ToTensor()
)

testing_data = datasets.FashionMNIST(
    root = 'Data',
    train = False,
    download = True,
    transform=ToTensor()
)