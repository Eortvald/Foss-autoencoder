import numpy as np
import torch, torchvision
from torch import nn
import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from os import listdir
from data.MyDataset import *

PATH_dict = {
    '10K': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
    'gamer': 'C:/Data/DeepEye/Foss_student/tenkblobs/',
    '224' : 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenhblobsA/',
    'validation' : 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/',
    'mix' : 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/grainmix'
}

path = PATH_dict['validation']

m = np.load('../10K_mean.npy')
s = np.load('../10K_std.npy')

print(f'mean: {m}\n std:{s}')


S = transforms.Compose([Mask_n_pad(H=180, W=80)])
Dataset = KornDataset(data_path=path, transform=S, label_path=None)

print(Dataset[8][0])

STATloader = DataLoader(Dataset, batch_size=1000, num_workers=0)

Tens = transforms.ToTensor()

means = []
stds = []


for inputs, label in STATloader:
    #print(inputs[0][0][80:90])
    temp_mean = torch.mean(inputs, dim=(0, 2, 3))
    print(temp_mean)
    temp_std = torch.std(inputs, dim=(0, 2, 3))

    means.append(temp_mean)
    stds.append(temp_std)



mean = torch.mean(means, dim=(0, 2, 3))
std = torch.std(stds, dim=(0, 2, 3))

np.save('../MEAN', mean)
np.save('../STD', std)

print(mean)
print(std)





"""

PATH_dict = {'10K': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
        'gamer': 'C:/Data/DeepEye/Foss_student/tenkblobs/'}

### If training on Foss Laptop select '10K'
### If training on Gamer select 'gamer'
PATH = PATH_dict['gamer']

def stat_npy_dir(path: str, subset: str):
    path = path + subset


    data_x = []
    data_y = []

    folder = listdir(path)
    folder_images = len(folder)
    for i, NPY in enumerate(folder):

        if i % 10 == 0:

            print(f'Images loaded: [{i}/{folder_images}]  ------  {str(datetime.datetime.now())[11:-7]}')

        img, _ = np.load(path+NPY, allow_pickle=True)
        data_x.append(img)


    print(f'Done reading {subset} images')
    wx = torch.tensor(data_x, dtype=torch.float)
    print('Begining permute')
    tx = wx.permute(0, 3, 1, 2)
    print('Finished permute')

    mean = torch.mean(tx, dim=(0, 2, 3))
    std = torch.std(tx, dim=(0, 2, 3))

    np.save('10K_mean', mean)
    np.save('10K_std', std)

    print(mean)
    print(std)


stat_npy_dir(PATH,'train/')

#xtrain = torch.normal(mean=10, std=2, size=(100, 8, 10, 10))
"""