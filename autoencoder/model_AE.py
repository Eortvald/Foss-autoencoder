# autoencoder
# Structuring of classes and functions inspired by AntixK - Github

import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# The device will be set to CUDA GPU if available,
# if not - it will de set to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

DOWNLOAD = False

b_size = 4

trainset = datasets.CIFAR10(root='./data', train=True, download=DOWNLOAD, transform=transform)
trainloader = DataLoader(trainset, batch_size=b_size, shuffle=False, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=2)

num_epochs = 10
learning_rate = 1e-3

# dimension of the hidden layers
layer_channels = [8, 16, 32, 64, 128]
z_dim = 20


class VAE(nn.Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()

        self.z_dim = z_dim

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),

        )

        self.LinC_mean = nn.Linear(6272, z_dim)
        self.LinC_var = nn.Linear(6272, z_dim)

        # Decoder setup
        self.decoder_link = nn.Linear(z_dim, 6272)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        encode_out = self.encoder(x)
        encode_out = encode_out.view(encode_out.size(0), -1)
        # print(encode_out.shape)

        mean = self.LinC_mean(encode_out)
        log_var = self.LinC_var(encode_out)

        # print(f'Mean Tensor: {mean.shape} \n Var Tensor: {log_var.shape}')

        return [mean, log_var]

    def reparameterize(self, mean, log_var):
        std = torch.exp(log_var * 0.5)
        epsilon = torch.randn_like(std)

        return epsilon * std + mean

    def decode(self, Z):
        # Dense layer
        Z = self.decoder_link(Z)
        # print(f'Z shape : {Z.shape}')
        Z = Z.view(Z.size(0), 128, 7, 7)

        # print(f'Z shape : {Z.shape}')

        # Reshape Linear into Conv ready shape again
        decode_out = self.decoder(Z)
        return decode_out

    def forward(self, x):
        # encode out put to mean and var

        mean, log_var = self.encode(x)

        # reparametize mean and var and to get Z (code layer)
        Z = self.reparameterize(mean, log_var)

        # decode z to get x_hat
        x_hat = self.decode(Z)

        return [x_hat, mean, log_var]

    def grain_from_Zspace(self, num_grain):
        generated_z = torch.randn(num_grain)
        generated_z = generated_z.to('cpu')
        generated_grains = self.decode(generated_z)

        return generated_grains

    def grain_hat(self, x):
        return self.forward(x)[0]


def loss_function(x_hat, x, mean, log_var):
    """
    Implementation of the ELBO - Evidence Lower Bound Loss metric
    :param x_hat: Decoder output
    :param x: Original Images
    :param mean: Mean from encoder
    :param log_var: Variance from encoder
    :return: Cross entropy + KL divergence between the two distributions q and p
    """
    # Cross Entropy
    # print(f'{x_hat}, \n MIN:{torch.min(x_hat)} \n MAX: {torch.max(x_hat)}')
    BCE = F.mse_loss(x_hat, x)

    # KL divergence
    KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim=1), dim=0)

    return BCE + KLD


model = VAE(z_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":

    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, (img, _) in enumerate(testloader):

            # forward
            output, mean, log_var = model(img)
            loss = loss_function(output, img, mean, log_var)

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
