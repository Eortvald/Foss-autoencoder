import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from datetime import *
from tqdm import tqdm
from os import listdir

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# PATHS
PATH_dict = {
    '10K': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
    'gamer': 'C:/Data/DeepEye/Foss_student/tenkblobs/',
    '224' : 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenhblobsA/',
    'validation' : 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/'
}

### If training on Foss Laptop select '10K'
### If training on Gamer select 'gamer'
PATH = PATH_dict['gamer']

# Transforms
MEAN_8ch = np.load('10K_mean.npy')
STD_8ch = np.load('10K_std.npy')

T = transforms.Normalize(mean=MEAN_8ch, std=STD_8ch)






# Folder loading

def npy_dir(path: str, subset: str):
    path = path + subset

    make_numeric = {'Oat': 1,
                    'Broken': 2,
                    'Rye': 3,
                    'Wheat': 4,
                    'BarleyGreen': 5,
                    'Cleaved': 6,
                    'Skinned': 7}

    data_x = []
    data_y = []

    folder = listdir(path)
    folder_images = len(folder)
    for i, NPY in enumerate(folder):

        if i % 200 == 0:
            print(f'Images loaded: [{i}/{folder_images}]  ------  {str(datetime.now())[11:-7]}')
        img, label = np.load(path + NPY, allow_pickle=True)
        data_x.append(img)
        numeric_label = make_numeric[label]
        data_y.append(numeric_label)

        # print(i,numeric_label)

    print(f'Done reading {subset} images')
    wx = torch.tensor(data_x, dtype=torch.float)
    print('Beginning permute')
    tx = wx.permute(0, 3, 1, 2)
    print('Finished permute')
    ty = torch.tensor(data_y, dtype=torch.long)
    print(f'Dimension of X is :{tx.size()}')
    print(tx[0])
    print('Beginning Norm transform')
    tx = T(tx)
    print('Norm transform finished')
    print(f'Dimension of X is :{tx.size()}------------')
    print(tx[0])
    return tx, ty


xtrain, ytrain = npy_dir(PATH, 'train/')
Ktrain_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=100, num_workers=0, shuffle=True)

xtest, ytest = npy_dir(PATH, 'test/')
Ktest_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, num_workers=0, shuffle=True)

if __name__ == "__main__":
    pass

    # for b_index, (X, y) in enumerate(Ktest_loader):
    #   if b_index % 10 == 0:
    #      print(f'Batch number:{b_index} | Image:{X[0]}\n Image Label:{y[0]}')
