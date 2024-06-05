import torch
from torch import nn
from torch.utils.data import DataLoader
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

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(testing_data, batch_size = batch_size)