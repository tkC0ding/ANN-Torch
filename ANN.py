import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
    root = 'Data',
    train = True,
    download = True,
    transform = ToTensor()
)

testing_data = datasets.MNIST(
    root = 'Data',
    train = False,
    download = True,
    transform = ToTensor()
)

batch_size = 64

train_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(testing_data, batch_size=batch_size)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(784, 128)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(128, 64)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(64, 16)
        self.r3 = nn.ReLU()
        self.l4 = nn.Linear(16, 10)

        nn.ModuleList([self.flatten, self.l1, self.r1, self.l2, self.l3, self.l4, self.r2, self.r3])
    
    def forward(self, x):
        out = self.flatten(x)
        out = self.l1(out)
        out = self.r1(out)
        out = self.l2(out)
        out = self.r2(out)
        out = self.l3(out)
        out = self.r3(out)
        out = self.l4(out)
        return(out)

model = DNN().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(model)

print(f"Using {device} device.")

def train(model, loss_fn, optimizer, dataloader):
    acc = 0
    for X,y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        acc += (y_pred.argmax(1) == y).type(torch.float).sum().item()/len(y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = acc/len(dataloader)
    return((loss.item(), acc))

def test(model, loss_fn, dataloader):
    acc = 0
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            acc += (pred.argmax(1) == y).type(torch.float).sum().item()/len(y)
        acc = acc/len(dataloader)
        return((loss.item(), acc))

num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, loss_fn, optimizer, train_loader)
    val_loss, val_acc = test(model, loss_fn, test_loader)
    print(f"Epoch:{epoch+1} loss:{train_loss} acc:{train_acc} val_loss:{val_loss} val_acc:{val_acc}")

torch.save(model.state_dict(), 'SavedModel/model.pth')