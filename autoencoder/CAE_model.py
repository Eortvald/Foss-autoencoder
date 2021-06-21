#Convolutional Autoencoder
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from data.MyDataset import *
from torch.utils.data import DataLoader
from torchinfo import *
from model_run import to_img, save_images



class CAE(nn.Module):

    def __init__(self, z_dim, h_dim: list = [12, 18, 24]):
        super(CAE, self).__init__()


        #================================ Encoder ==============================#
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=5, stride=3, padding=0, bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.Conv2d(12, 18, kernel_size=4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(18),
            nn.ReLU(True),
            nn.Conv2d(18, 24, kernel_size=4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.Flatten()
        )

        # ============ Z layer ============#
        self.fcZ = nn.Linear(2016,z_dim)


        #================================ Decoder ==============================#
        self.decoder_brigde = nn.Linear(z_dim, 2016)
        self.decoder = nn.Sequential(
            nn.Unflatten(1,(24,14,6)),
            nn.ConvTranspose2d(24, 18, kernel_size=4, stride=2, padding=1, output_padding=1, bias = False),
            nn.BatchNorm2d(18),
            nn.ReLU(True),
            nn.ConvTranspose2d(18, 12, kernel_size=4, stride=2, padding=1, output_padding=[1,0], bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 8, kernel_size=5, stride=3, padding=0, output_padding=[1,0]),
            nn.Tanh()
        )

    def encode(self, x):
        encode_out = self.encoder(x)
        Z = self.fcZ(encode_out)
        return Z

    def decode(self, Z):
        Z = self.decoder_brigde(Z)
        decode_out = self.decoder(Z)
        return decode_out

    def forward(self, x):
        Z = self.encode(x)
        X_hat = self.decode(Z)
        return X_hat


if __name__ == '__main__':

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

    device = 'cpu'

    aemodel = CAE(z_dim=30)
    aemodel.load_state_dict(torch.load('../autoencoder/model_dicts/PTH_Grain/CAE_69.pth', map_location=device))
    aemodel.eval()

    TFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    ENCO = lambda img: aemodel.encode(img)

    traindata = KornDataset(data_path=PATH + '/train/', transform=TFORM, label_path=label_path)

    X, label = traindata[345]
    X.to(device)
    X_hat = aemodel(X)

    save_images(X, X_hat, show=True)








    #summary(model, input_size=(batch_size, 8, 180, 80), col_names=["input_size", "output_size","kernel_size"])
