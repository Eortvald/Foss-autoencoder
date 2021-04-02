# autoencoder
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

# The tensors will be moved to CUDA GPU if available,
# if not - it will stay on CPU
torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

class VAE(nn.Module):

    def __init__(self, in_: int,):
        super(VAE, self).__init__()

        # Encoder setup
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1))

        # Decoder setup
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 10, stride=2, padding=1),
            nn.Tanh()
        )


    def encode(self):
        pass

    def decode(self):
        pass

    def reparameterize(self):
        pass

    def forward(self):

        # encode out put to mean and var

        # reparametize mean and var and to get Z (code layer)

        #decode z
        pass

    def loss_function(self):
        # ELBO
        pass

    def sample(self):
        pass

    def generate(self):
        pass


# Tensors to Numpy array
# Tensor.numpy()

if __name__ == "__main__":
    VariationalAE = VAE()

    # Script in under this statement will only be run when this file is executed
    # If you import and run this file from another script,
    # the interpreter will ignore function call made in this statement
