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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        probabilities = self.net(x)
        return(probabilities)

model = Net().to(device)
print(model)