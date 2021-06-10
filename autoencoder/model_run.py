import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from datetime import *
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from CAE_model import *
from data.dataload_collection import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

# Train function

def train_AE(model, train_loader):
    model.train()
    train_loss = 0
    dataset_size = len(train_loader.dataset)

    for batch_num, (X, _) in enumerate(train_loader):
        # Regeneration and loss
        X = X.to(device)
        optimizer.zero_grad()
        X_hat = model(X)

        loss = criterion(X_hat, X)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_num % 10 == 0:
            print(100 * '~')
            progress = batch_num * len(X)
            print(f'Batch[{batch_num}] - Batch loss:{loss.item()}\t | Avg. per. Image Loss: {loss.item() / len(X)}) [{progress}/{dataset_size}]')

    train_loss /= dataset_size
    print(f'\t \t Train Error: Avg loss: {train_loss}')
    print(100 * '^')
    return train_loss


def test_AE(model, test_loader):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (X, _) in enumerate(test_loader):
            X = X.to(device)
            X_hat = model(X)
            test_loss += criterion(X_hat, X).item()

    test_loss /= len(test_loader.dataset)
    print(f'\t \t \t Test Error: Avg loss: {test_loss} \n')

    if epoch == 100:
        pic = to_img(X_hat.cpu().X)
        save_image(pic, './img/image_{}.png'.format(epoch))

    return test_loss


if __name__ == "__main__":

    # b_size =
    num_epochs = 100
    learning_rate = 1e-3

    # dimension of the hidden layers
    # layer_channels = [8, 16, 32]
    z_dim = 30

    CAE_10Kmodel = CAE(z_dim=z_dim)
    CAE_10Kmodel = CAE_10Kmodel.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(CAE_10Kmodel.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_log = []
    test_log = []

    for epoch in range(num_epochs):
        print(f'\n\t\t------------------------------Epoch: {epoch + 1}------------------------------')
        train_save = train_AE(CAE_10Kmodel, Ktrain_loader)
        train_log.append(train_save)

        print('\t\t\t>>>>>>>>>>>>>>>>TEST RESULTS<<<<<<<<<<<<<<<<<')
        test_save = test_AE(CAE_10Kmodel, Ktest_loader)
        test_log.append(test_save)


    print(f'Length of train log: {len(train_log)}')
    print(f'Length of test log: {len(test_log)}')
    print(train_log)


    torch.save(CAE_10Kmodel.state_dict(), 'model_dicts/CAE_10Kmodel.pth')



    #np.save('model_dicts/train_results', train_save, allow_pickle=False)
    # np.save('model_dicts/test_results', test_save, allow_pickle=False)

    plt.plot(np.arange(len(train_log)), train_log, label='Train')
    plt.plot(np.arange(len(test_log)), test_log, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train vs Test")
    plt.legend()
    abspath = 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/Foss-autoencoder/plots/autoencoder-plots/'
    plt.savefig(f"{abspath}CAE_10K_Results-{str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')}.png")

