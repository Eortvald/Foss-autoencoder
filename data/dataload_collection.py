import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# PATHS
PATH_10K = 'path'

# Transforms
Norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#Folder loading


def dataset_from_collection(PATH:str, transform, batch_size: int, filetype: str='.npy'):

    npy_load = lambda PATH:torch.from_numpy(np.load(PATH,allow_pickle=True))

    train_data = datasets.DatasetFolder(root=PATH+'/train', loader=npy_load, extensions=[filetype], transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size,num_workers=2 )

    test_data = datasets.DatasetFolder(root=PATH+'/test', loader=npy_loader, extensions=(ext))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

    return train_loader, test_loader

dataset_from_collection()

#Transforms


#Loadder delegation
