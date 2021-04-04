# autoencoder
# Structuring of classes and functions inspired by AntixK - Github

import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# The tensors will be moved to CUDA GPU if available,
# if not - it will stay on CPU
torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

DOWNLOAD = False

trainset = datasets.CIFAR10(root='./data', train=True, download=DOWNLOAD, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)


# testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

#dimension of the hidden layers
layer_channels = [8,16,32,64,128]
z_dim = 64

class VAE(nn.Module):

    def __init__(self, x_channels, z_dim):
        super(VAE, self).__init__()

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Flatten()
        )

        self.z_dim = z_dim
        self.LinC_mean = nn.Linear(1024, z_dim)
        self.LinC_var = nn.Linear(1024, z_dim)

        # Decoder setup
        self.decoder_link = nn.Linear(z_dim, 1024)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1),
            nn.Tanh
        )

    def encode(self, x):

        encode_out = self.encoder(x)

        mean = self.LinC_mean(encode_out)
        log_var = self.LinC_var(encode_out)

        return [mean, log_var]

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)

        return mean + epsilon * std

    def decode(self, Z):
        #
        Z = self.decoder_link(Z)

        # Fold the flatten tensor back in to shape
        Z = Z.view(-1, 256, 2, 2)
        decode_out = self.decoder(Z)
        return decode_out

    def forward(self, x):
        # encode out put to mean and var
        mean, log_var = self.encoder(x)

        # reparametize mean and var and to get Z (code layer)
        Z = self.reparameterize(mean, log_var)

        # decode z to get x_hat
        x_hat = self.decode(Z)

        return [x_hat, x, mean, log_var]


    def loss_function(self):
        # ELBO
        pass

    def generate_grain(self, num_grain):

        generated_z = torch.randn(num_grain)
        generated_z = generated_z.to('cpu')
        generated_grains = self.decode(generated_z)

        return generated_grains

    def grain_(self,x):

        return self.forward(x)[0]

model = VAE()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)



if __name__ == "__main__":
    pass



    # Script in under this statement will only be run when this file is executed
    # If you import and run this file from another script,
    # the interpreter will ignore function call made in this statement
