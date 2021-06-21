import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from torch import nn, optim
from data.MyDataset import *
from autoencoder.CAE_model import *
import os
def to_img(img):
    img = img.cpu()[0]
    img = img * STD[:, None, None] + MEAN[:, None, None]
    img = np.int64(img.numpy().transpose(1, 2, 0))
    img = np.dstack((img[:, :, 4], img[:, :, 2], img[:, :, 1]))

    return img
epoch = 1

def save_images(x, xhat, show=False):
    x = to_img(x)
    xhat = to_img(xhat)

    fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(4, 5))

    ax1.imshow(x)
    ax1.set_title(f'Input Image ({epoch})')

    ax2.imshow(xhat)
    ax2.set_title(f'Reconstructed ({epoch})')

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig1.savefig(f"../plots/autoencoder-plots/images/img_{epoch}.png")
    plt.close(fig1)
    if show:
        plt.show()



MEAN = np.load('../MEAN.npy')
STD = np.load('../STD.npy')

label_path = '../preprocess/classifier_labels.csv'
PATH_dict = {
    '10K_remote': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
    '10K_gamer': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/tenkblobs/',
    '224': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenhblobsA/',
    'validation_grain': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/',
    'validation_blob': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_blob/',
    'grainmix': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/grainmix/'
}
labels = ['Oat','Broken', 'Rye', 'Wheat', 'BarleyGreen','Cleaved', 'Skinned','Barley']
DATA_SET = 'validation_blob'
PATH = PATH_dict[DATA_SET]

TFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])

traindata = KornDataset(data_path=PATH+'/train/', transform=TFORM, label_path=label_path)
trainload = DataLoader(traindata, batch_size=100, shuffle=True, num_workers=0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
aemodel = CAE(z_dim=30).to(device)
aemodel.load_state_dict(torch.load('model_dicts/CAE_10Kmodel.pth', map_location=device))
aemodel.eval()

AUTOENCODE = lambda img: aemodel(img)

savepath = '../plots/autoencoder-plots/reconstructions/'


for i, (img, label) in enumerate(trainload):
    J = np.random.randint(1,100)
    img, label = traindata[J]

    recon = AUTOENCODE(img)
    save_images(img, recon, show=True)