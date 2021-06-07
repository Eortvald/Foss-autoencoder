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

    def __init__(self, z_dim, h_dim: list = [12, 18, 24]):
        super(CAE, self).__init__()

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=0, bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.Conv2d(12, 18, 3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(18),
            nn.ReLU(True),
            nn.Conv2d(18, 24, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.Flatten()
        )

        self.fcZ = nn.Linear(26400,z_dim)

        # Decoder setup
        self.decoder = nn.Sequential(
            nn.Unflatten(1,(30,40,40)),
            nn.ConvTranspose2d(24, 18, 2, stride=1, padding=0, bias = False),
            nn.BatchNorm2d(18),
            nn.ReLU(True),
            nn.ConvTranspose2d(18, 12, 3, stride=2, padding=0, bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 8, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        encode_out = self.encoder(x)
        print(f'encode dim:{encode_out.size()}')
        #flat_encode_out = encode_out.view(encode_out.size(0), -1)
        Z = self.fcZ(encode_out)
        print(f'Z dimension: {Z.size()}')
        return Z

    def decode(self, Z):

        decode_out = self.decoder(Z)
        return decode_out

    def forward(self, x):
        Z = self.encode(x)
        X_hat = self.decode(Z)
        return X_hat




if __name__ == "__main__":
    pass
    # Script in under this statement will only be run when this file is executed
    # If you import and run this file from another script,
    # the interpreter will ignore function call made in this statement