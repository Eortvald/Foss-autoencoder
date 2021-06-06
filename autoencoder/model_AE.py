# autoencoder
# Structuring of classes and functions inspired by AntixK - Github

import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_run import *

# The device will be set to CUDA GPU if available,
# if not - it will de set to CPU


train_loader = 2
test_loader = 2
b_size = 4
num_epochs = 10
learning_rate = 1e-3

# dimension of the hidden layers
layer_channels = [8, 16, 32]
z_dim = 30


class AE(nn.Module):

    def __init__(self, z_dim):
        super(AE, self).__init__()

        self.z_dim = z_dim

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

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
        encode_out = encode_out.view(encode_out.size(0), -1)
        return [mean, log_var]


    def decode(self, Z):
        Z = Z.view(Z.size(0), 128, 7, 7)
        decode_out = self.decoder(Z)
        return decode_out

    def forward(self, x):
        X_hat = self.decode(self.encode(x))
        return X_hat



def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    size = len
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs
def test()

model = AE(z_dim)

if __name__ == "__main__":

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, (img, _) in enumerate(trainloader):

            # forward
            output, mean, log_var = model(img)
            loss = criterion()

            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        to_print = f"epoch:{epoch + 1} of {num_epochs} | Loss: {loss.item() / b_size}  | {loss.item()}"
        print(to_print)
    torch.save(model.state_dict(), 'VAE.pth')
    # Script in under this statement will only be run when this file is executed
    # If you import and run this file from another script,
    # the interpreter will ignore function call made in this statement
