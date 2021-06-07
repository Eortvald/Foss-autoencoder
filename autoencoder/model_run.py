import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from CAE_model import CAE
from data.dataload_collection import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


# Train function

def train_AE(model, train_loader, loss_func, opt_method):
    model.train()
    size = len(train_loader.dataset)
    train_loss = 0
    print(f'Training on {device}')

    for batch_num, (X, _) in enumerate(train_loader):
        # Regeneration and loss
        X = X.to(device)
        X_hat = model(X)
        loss = criterion(X_hat, X)

        # Back prop
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_num % 100 == 0:
            loss, progress = loss.item(), batch_num * len(X)
            print(f'loss:{loss} ({loss / len(X)})  [{progress / size}]')

    print(f'Train Error - Avg loss: {train_loss / size}')


def test_AE(model, test_loader):
    model.eval()
    size = len(test_loader.dataset)
    test_loss = 0

    with torch.no_grad():
        for i, (X, _) in enumerate(test_loader):
            X = X.to(device)
            X_hat = model(X)
            test_loss += criterion(X_hat, X).item()

    test_loss /= size

    print(f'Test Error: Avg loss: {test_loss} \n')


model = CAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
num_epochs = 10

if __name__ == "__main__":

    for epoch in range(num_epochs):
        print(f'Epoch {epoch} \n---------------------')

        train_AE(model, train_loader)
        test_AE(model, test_loader)

    torch.save(model.state_dict(), 'model_dicts/VAE.pth')
