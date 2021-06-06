import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_run import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

#Train function


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