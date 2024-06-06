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

num_epochs = 10

print(f"Using {device} device")
for epoch in range(num_epochs):
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch : {epoch + 1} Loss : {loss.item()}")
print("training is done!")

num_batches = len(test_loader)
size = len(test_loader.dataset)

test_loss = 0
correct = 0

with torch.no_grad():
    for X,y in test_loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        test_loss += loss_fn(pred, y).item()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    avg_loss = test_loss/num_batches
    avg_acc = correct/size
    print(f"Accuracy : {avg_acc} Loss : {avg_loss}")

torch.save(model.state_dict(), 'SavedModel/model.pth')