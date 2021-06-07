import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# PATHS
PATH_dict = {'10K': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs',
         'nyt datasæt': 'path til nyt datasæt'}

# Transforms
Norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


# Folder loading

def dataset_from_collection(PATH:str, batch_size: int, transform: object = None, filetype: str = '.npy'):

    npy_load = lambda PATH: torch.from_numpy(np.load(PATH, allow_pickle=True))

    train_data = datasets.DatasetFolder(root=PATH + '/train', loader=npy_load, extensions=[filetype],
                                        transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2)

    test_data = datasets.DatasetFolder(root=PATH + '/test', loader=npy_load, extensions=[filetype],transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

    return train_loader, test_loader


train_loader, test_loader = dataset_from_collection(PATH=PATH_dict['10K'], transform=Norm_transform, batch_size=128)

for b_index, (X,y) in enumerate(test_loader):
    if b_index % 10 == 0:
        print(f'Batch number:{b_index} | Image:{X}\n Image Label:{y}')
# Transforms


# Loadder delegation
