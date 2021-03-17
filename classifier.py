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





