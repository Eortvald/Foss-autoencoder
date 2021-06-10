#Convolutional Autoencoder
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import *

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
        self.fcZ = nn.Linear(2688,z_dim)


        #================================ Decoder ==============================#
        self.decoder_bridge = nn.Linear(z_dim, 2688)
        self.decoder = nn.Sequential(
            nn.Unflatten(1,(24,16,7)),
            nn.ConvTranspose2d(24, 18, 4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(18),
            nn.ReLU(True),
            nn.ConvTranspose2d(18, 12, 4, stride=2, padding=[0,1], output_padding=[0,1], bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 8, 5, stride=3, padding=[1,0], output_padding=[2,0]),
            nn.Tanh()
        )

    def encode(self, x):
        encode_out = self.encoder(x)
        Z = self.fcZ(encode_out)
        return Z

    def decode(self, Z):
        Z = self.decoder_bridge(Z)
        decode_out = self.decoder(Z)
        return decode_out

    def forward(self, x):
        Z = self.encode(x)
        X_hat = self.decode(Z)
        return X_hat
