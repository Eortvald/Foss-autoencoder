import numpy as np
from datetime import *
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from CAE_model import *
from data.dataload_collection import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Train function
train_stat = []
test_stat = []


def train_AE(model, train_loader):
    model.train()
    size = len(train_loader.dataset)
    # print(f'train size:{size}')
    train_loss = 0
    t_loss = 0
    # print(f'Training on {device}')

    for batch_num, (X, _) in enumerate(train_loader):
        # Regeneration and loss
        _ = _.to(device)
        X = X.to(device)
        X_hat = model(X).to(device)
        X_hat = X_hat.to(device)
        loss = criterion(X_hat, X)

        # Back prop
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        t_loss += loss.item() * X.size(0)

        train_stat.append(loss)
        if batch_num % 5 == 0:
            loss, progress = loss.item(), batch_num * len(X)
            print(f'Batch[{batch_num}] | loss:{loss} ({loss/len(X)}) [{progress}/{size}]')


    print(f'Train Error - Avg loss: {train_loss / size}-----------## {t_loss/len(train_loader)}')


def test_AE(model, test_loader):
    model.eval()
    size = len(test_loader.dataset)
    # print(f'test size:{size}')
    test_loss = 0

    test_out = []
    with torch.no_grad():
        for i, (X, _) in enumerate(test_loader):
            _ = _.to(device)
            X = X.to(device)
            X_hat = model(X).to(device)
            test_loss += criterion(X_hat, X).item()
            loss = criterion(X_hat, X).item()

            test_stat.append(loss)

    test_loss /= size
    print(f'Test Error: Avg loss: {test_loss} \n')


if __name__ == "__main__":

    # b_size =
    num_epochs = 50
    learning_rate = 1e-3

    # dimension of the hidden layers
    # layer_channels = [8, 16, 32]
    z_dim = 100

    CAE_10Kmodel = CAE(z_dim=z_dim)
    CAE_10Kmodel = CAE_10Kmodel.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(CAE_10Kmodel.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} \n---------------------')
        train_AE(CAE_10Kmodel, Ktrain_loader)
        test_AE(CAE_10Kmodel, Ktest_loader)

        torch.save(CAE_10Kmodel.state_dict(), 'model_dicts/CAE_10Kmodel.pth')
"""
    train_save = np.array([loss.cpu().detach().numpy() for loss in train_stat])
    test_save = np.array(losst.cpu().detach().numpy() for losst in test_stat)
    np.save('model_dicts/train_results', train_save, allow_pickle=False)
    np.save('model_dicts/test_results', test_save, allow_pickle=False)

    torch.save(CAE_10Kmodel.state_dict(), 'model_dicts/CAE_10Kmodel.pth')

    plt.plot(np.arange(len(train_save)), train_save, label='train')  # etc.
    plt.plot(np.arange(len(test_save)), test_save, label='test')
    plt.xlabel('acummulated batches')
    plt.ylabel('Loss')
    plt.title("Train vs Test")
    plt.legend()
    plt.savefig(f"results_run-{str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')}.png")
"""