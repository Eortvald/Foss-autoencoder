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
from data.MyDataset import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


def to_img(img):
    img = img.cpu()[0]
    img = img * STD[:, None, None] + MEAN[:, None, None]
    img = np.int64(img.numpy().transpose(1, 2, 0))
    img = np.dstack((img[:, :, 4], img[:, :, 2], img[:, :, 1]))

    return img


def show_images(x, xhat):
    x = to_img(x)
    xhat = to_img(xhat)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(4, 9))

    ax1.imshow(x)
    ax1.set_title('Input Image')

    ax2.imshow(xhat)
    ax2.set_title('Reconstructed')

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f"../plots/autoencoder-plots/images/img_{epoch}-{str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')}.png")
    plt.show()


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
            print(
                f'Batch[{batch_num}] - Batch loss:{loss.item()}\t | Avg. per. Image Loss: {loss.item() / len(X)}) [{progress}/{dataset_size}]')

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

    if epoch % 10 == 0:
        show_images(X, X_hat)


    return test_loss


if __name__ == "__main__":

    MEAN = np.load('../MEAN.npy')
    STD = np.load('../STD.npy')

    label_path = '../preprocess/Classifier_labels.csv'
    PATH_dict = {
        '10K_remote': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
        '10K_gamer': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/tenkblobs/',
        '224': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenhblobsA/',
        'validation_grain': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/',
        'validation_blob': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_blob/',
        'grainmix': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/grainmix/'
    }

    DATA_SET = 'validation_grain'
    PATH = PATH_dict[DATA_SET]



    TFORM = transforms.Compose([Mask_n_pad(H=180, W=80), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])

    # Training conditions
    BSIZE = 100
    num_epochs = 100
    learning_rate = 1e-3
    w_decay = 1e-5

    # Bottleneck layer size
    z_dim = 30

    # Dataset Init
    traindata = KornDataset(data_path=PATH, transform=TFORM)  # the dataset object can be indexed like a regular list
    trainload = DataLoader(traindata, batch_size=BSIZE, shuffle=True, num_workers=0)
    train_log = []

    testdata = KornDataset(data_path=PATH, transform=TFORM)
    testload = DataLoader(testdata, batch_size=BSIZE, shuffle=True,  num_workers=0)
    test_log = []

    # Model Inite
    CAE_10Kmodel = CAE(z_dim=z_dim)
    CAE_10Kmodel = CAE_10Kmodel.to(device)

    # Loss and Backwards settings
    criterion = nn.MSELoss()
    optimizer = optim.Adam(CAE_10Kmodel.parameters(), lr=learning_rate, weight_decay=w_decay)




    # Training and test of model

    for epoch in range(num_epochs):
        print(f'\n\t\t------------------------------Epoch: {epoch + 1}------------------------------')
        train_save = train_AE(CAE_10Kmodel, trainload)
        train_log.append(train_save)

        print('\t\t\t>>>>>>>>>>>>>>>>TEST RESULTS<<<<<<<<<<<<<<<<<')
        test_save = test_AE(CAE_10Kmodel, testload)
        test_log.append(test_save)

    print(f'Length of train log: {len(train_log)}')
    print(f'Length of test log: {len(test_log)}')
    print(train_log)

    torch.save(CAE_10Kmodel.state_dict(), 'model_dicts/CAE_10Kmodel.pth')

    np.savez('model_dicts/session_results', train_log, test_log)
    np.save('model_dicts/test_results', test_log, allow_pickle=False)

    plt.plot(np.arange(len(train_log)), train_log, label='Train')
    plt.plot(np.arange(len(test_log)), test_log, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(axis='both', color='0.95')
    plt.title(f"Train vs Test \t - {DATA_SET} \nBatch size:{BSIZE} | z_dim:{z_dim} | Dataset size:{len(trainload.dataset)}" )
    plt.legend()
    plt.savefig(f"../plots/autoencoder-plots/CAE_10K_Results-{str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')}.png")
