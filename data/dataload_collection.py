import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# PATHS


# Transforms
Norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#Folder loading

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


def dataset_from_collection(trainPATH:str,testPATH: str, transform, batch_size: int, filetype: str='.npy'):

    train_data = datasets.DatasetFolder(root=trainPATH, loader=npy_loader, extensions=[filetype], transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size,num_workers=2 )

    test_data = datasets.DatasetFolder(root=testPATH, loader=npy_loader, extensions=(ext))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

#Transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Loadder delegation
