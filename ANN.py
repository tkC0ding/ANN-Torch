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

testing_data = datasets.MNIST(
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
        self.l1 = nn.Linear(784, 512)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(512, 512)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(512, 10)
        self.s1 = nn.Softmax()
    
    def forward(self, x):
        out = self.flatten(x)
        out = self.l1(out)
        out = self.r1(out)
        out = self.l2(out)
        out = self.r2(out)
        out = self.l3(out)
        out = self.s1(out)
        return(out)

model = Net().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)