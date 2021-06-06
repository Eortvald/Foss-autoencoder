import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

#Train function
model = AE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


def train_AE(model, train_loader, epoch):
    model.train()
    size = len(train_loader.dataset)
    train_loss = 0

    for b_index, (X, _) in enumerate(train_loader):
        X = X.to(device)
        print(f'Training data has been moved to{device}')
        optimizer.zero_grad()
        X_recon = model(X)
        loss = criterion(X_recon,X)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        if b_index % ...:
            print(f'Train Epoch {epoch} '
                  f'[{b_index * len(X)}/{size} ({100 * b_index/len(train_loader)}%)]'
                  f'\tLoss: {loss.item() / len(X)}')
            #to_print = f"epoch:{epoch + 1} of {num_epochs} | Loss: {loss.item() / b_size}  | {loss.item()}"
            #print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
    print(f'====> Epoch:{epoch}  Average loss:{train_loss/size}')

def test_AE(model, train_loader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (X, _) in enumerate(test_loader):
            X = X.to(device)




    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    train_AE()
    test_AE()

    torch.save(model.state_dict(), 'VAE.pth')