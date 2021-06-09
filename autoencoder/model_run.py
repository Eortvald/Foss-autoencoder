import numpy as np
from datetime import *
from tqdm import tqdm
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


def train_AE(model, train_loader):
    model.train()
    size = len(train_loader.dataset)
    # print(f'train size:{size}')
    train_loss = 0
    # print(f'Training on {device}')

    for batch_num, (X, _) in tqdm(enumerate(train_loader)):
        # Regeneration and loss
        X = X.to(device)
        optimizer.zero_grad()

        X_hat = model(X)
        loss = criterion(X_hat, X)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_num % 10 == 0:
            loss, progress = loss.item(), batch_num * len(X)
            print(f'Batch[{batch_num}] | loss:{loss} ({loss/len(X)}) [{progress}/{size}]')
            print(10*'##')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_num * len(X), len(train_loader.dataset),
                       100. * batch_num / len(train_loader),
                       loss.item() / len(X)))

    train_loss /= size
    print(f'Train Error: Avg loss: {train_loss}')

    print(10 * '##')
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test_AE(model, test_loader):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (X, _) in tqdm(enumerate(test_loader)):
            X = X.to(device)
            X_hat = model(X)
            test_loss += criterion(X_hat, X).item()
            loss = criterion(X_hat, X).item()

    test_loss /= len(test_loader.dataset)
    print(f'Test Error: Avg loss: {test_loss} \n')


if __name__ == "__main__":

    # b_size =
    num_epochs = 100
    learning_rate = 1e-3

    # dimension of the hidden layers
    # layer_channels = [8, 16, 32]
    z_dim = 60

    CAE_10Kmodel = CAE(z_dim=z_dim)
    CAE_10Kmodel = CAE_10Kmodel.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(CAE_10Kmodel.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} \n---------------------')
        train_AE(CAE_10Kmodel, Ktrain_loader)
        test_AE(CAE_10Kmodel, Ktest_loader)

        torch.save(CAE_10Kmodel.state_dict(), 'model_dicts/CAE_10Kmodel.pth')


    torch.save(CAE_10Kmodel.state_dict(), 'model_dicts/CAE_10Kmodel.pth')

    #np.save('model_dicts/train_results', train_save, allow_pickle=False)
    #np.save('model_dicts/test_results', test_save, allow_pickle=False)

    #plt.plot(np.arange(len(train_save)), train_save, label='train')  # etc.
    #plt.plot(np.arange(len(test_save)), test_save, label='test')
    #plt.xlabel('acummulated batches')
    #plt.ylabel('Loss')
    #plt.title("Train vs Test")
    #plt.legend()
    #plt.savefig(f"results_run-{str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')}.png")
