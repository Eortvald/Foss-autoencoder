import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Folder loading

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


traindataset = datasets.DatasetFolder(root='PATH', loader=npy_loader,extensions=['.npy'])

#Transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Loadder delegation
