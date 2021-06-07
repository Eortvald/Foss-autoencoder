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

def train_AE(model, train_loader):
    model.train()
    size = len(train_loader.dataset)
    print(f'train size:{size}')
    train_loss = 0

    train_out = []
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

        if batch_num % 5 == 0:
            loss, progress = loss.item(), batch_num * len(X)
            print(f'loss:{loss} ({loss / len(X)})  [{progress / size}]')
            train_out.append(loss)
    print(f'Train Error - Avg loss: {train_loss / size}')

    return train_out


def test_AE(model, test_loader):
    model.eval()
    size = len(test_loader.dataset)
    print(f'test size:{size}')
    test_loss = 0

    test_out = []

    with torch.no_grad():
        for i, (X, _) in enumerate(test_loader):
            X = X.to(device)
            X_hat = model(X)
            test_loss += criterion(X_hat, X).item()
            loss = criterion(X_hat, X).item()

            if i % 5 == 0:
                test_out.append(loss)
    test_loss /= size
    print(f'Test Error: Avg loss: {test_loss} \n')

    return test_out


if __name__ == "__main__":

    # b_size =
    num_epochs = 10
    learning_rate = 1e-3

    # dimension of the hidden layers
    # layer_channels = [8, 16, 32]
    z_dim = 30

    CAE_10Kmodel = CAE(z_dim=z_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(CAE_10Kmodel.parameters(), lr=learning_rate)
    train_result = []
    test_result = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch} \n---------------------')

        train_stat = train_AE(CAE_10Kmodel, Ktrain_loader)
        train_result.append(train_stat)

        test_stat = test_AE(CAE_10Kmodel, Ktest_loader)
        test_result.append(test_stat)

        torch.save(CAE_10Kmodel.state_dict(), 'model_dicts/CAE_10Kmodel.pth')

    to_save = np.array([train_result, test_result])
    np.save('model_dicts/loss_results', to_save)
    torch.save(CAE_10Kmodel.state_dict(), 'model_dicts/CAE_10Kmodel.pth')

    plt.plot(np.arange(100), train_result, label='train')  # etc.
    plt.plot(np.arange(100), test_result, label='test')
    plt.xlabel('acummulated batches')
    plt.ylabel('Loss')
    plt.title("Train vs Test")
    plt.legend()
