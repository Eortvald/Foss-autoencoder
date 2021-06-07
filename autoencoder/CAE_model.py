#Convolutional Autoencoder

import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_run import *


class CAE(nn.Module):

    def __init__(self, in_channels, z_dim, h_dim: list = [12, 18, 24]):
        super(CAE, self).__init__()


        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,out , kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.fcZ = nn.Linear(4*32,z_dim)
        # Decoder setup
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 24, 2, stride=1, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 16, 3, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        encode_out = self.encoder(x)
        flat_encode_out = encode_out.view(encode_out.size(0), -1)
        Z = self.fcZ(flat_encode_out)
        return Z

    def decode(self, Z):
        Z = Z.view(Z.size(0), 128, 7, 7)

        decode_out = self.decoder(Z)
        return decode_out

    def forward(self, x):
        Z = self.encode(x)
        X_hat = self.decode(Z)
        return X_hat




if __name__ == "__main__":

    # Script in under this statement will only be run when this file is executed
    # If you import and run this file from another script,
    # the interpreter will ignore function call made in this statement
