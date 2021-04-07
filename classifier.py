# classifier
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root= "data",
    train = True,
    download = True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = False,
    transform=ToTensor()
)




if __name__ == "__main__":
    # Script in under this statement will only be run when this file is executed
    # If you import and run this file from another script,
    # the interpreter will ignore function call made in this statement
