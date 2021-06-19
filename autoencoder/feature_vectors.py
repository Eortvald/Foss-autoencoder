import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch, torchvision
import pandas as pd
import seaborn as sns
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from CAE_model import *
from data.MyDataset import *
import torch, torchvision
from datetime import *
import autoencoder, classifier, preprocess

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MEAN = np.load('../MEAN.npy')
STD = np.load('../STD.npy')

label_path = '../preprocess/Classifier_labels.csv'
PATH_dict = {
    '10K_remote': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
    '10K_gamer': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/tenkblobs/',
    '224': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenhblobsA/',
    'validation_grain': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/',
    'validation_blob': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_blob/',
    'grainmix': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/grainmix/'
}

DATA_SET = 'validation_blob'
PATH = PATH_dict[DATA_SET]

# Autoencoder setup
aemodel = CAE(z_dim=30).to(device)
aemodel.load_state_dict(torch.load('model_dicts/CAE_10Kmodel.pth', map_location=device))
aemodel.eval()
ENCO = lambda x: aemodel.encode(x)

TFORM = transforms.Compose([Mask_n_pad(H=180, W=80), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])

traindata = KornDataset(data_path=PATH, transform=TFORM,
                        label_path=label_path)
trainload = DataLoader(traindata, batch_size=2, shuffle=True, num_workers=0)

for i in range(100):
    img, label = traindata[i]









feature_class_vectors = {'Oat': 1,
                         'Broken': 2,
                         'Rye': 3,
                         'Wheat': 4,
                         'BarleyGreen': 5,
                         'Cleaved': 6,
                         'Skinned': 7,
                         'Barley*': 8}




tips = sns.load_dataset("tips")
tips.pop("tip")
tips.pop("smoker")
tips.pop("time")
tips.pop("size")
print(tips.head())
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.catplot(x='day', y='total_bill', row='sex', hue='sex', data=tips, kind='bar', dodge=False, saturation=.5,
                ci=None, aspect=0.9)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
g.fig.set_figheight(7)
g.fig.set_figwidth(10)

# def axla(sex):
#     plt.gca().text(-.02, .2, "sex", fontweight="bold",
#             ha="right", va="center", transform=plt.gca().transAxes)
# g.map(axla,"sex")


g.fig.subplots_adjust(hspace=0.2)
# g.set_titles(" ")

g.set(yticks=[])
g.despine(bottom=True, left=True)
plt.show()
