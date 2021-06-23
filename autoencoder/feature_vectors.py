import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch, torchvision
import pandas as pd
import seaborn as sns
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data.MyDataset import *
import torch, torchvision
from datetime import *
from classifier.MLP_Classifier import *

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
labels = ['Oat','Broken', 'Rye', 'Wheat', 'BarleyGreen','Cleaved', 'Skinned','Barley']
DATA_SET = 'validation_blob'
PATH = PATH_dict[DATA_SET]
hidden_out = [16, 12, 10]
# Autoencoder setup
ANN_model = ANN(30, hidden_out).to(device)
ANN_model.load_state_dict(torch.load('../classifier/model_dicts/ANN_Big.pth', map_location=device))
ANN_model.eval()
aemodel = CAE(z_dim=30).to(device)
aemodel.load_state_dict(torch.load('../autoencoder/model_dicts/PTH_Grain/CAE_69.pth', map_location=device))
aemodel.eval()


ENCO = lambda img: aemodel.encode(img)

TFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])

traindata = KornDataset(data_path=PATH+'/train/', transform=TFORM, label_path=label_path)
trainload = DataLoader(traindata, batch_size=1000, shuffle=True, num_workers=0)
criterion = nn.CrossEntropyLoss()

array = np.zeros((7, 7))
print(array)


for i, (img, label) in enumerate(trainload):
    label = label.to(device)
    inputs = img.to(device)
    inputs = ENCO(inputs)
    yhat = ANN_model(inputs)
    print(i)

    loss = criterion(yhat, label)

    _, yhat = torch.max(yhat, 1)

    for la, pre in zip(label.detach().cpu().numpy(), yhat.detach().cpu().numpy()):

        array[int(la - 1)][int(pre - 1)] += 1.



df_cm = pd.DataFrame(array, index=['Oat', 'Broken', 'Rye', 'Wheat', 'BarleyGreen', 'Cleaved', 'Skinned'],
                     columns=['Oat', 'Broken', 'Rye', 'Wheat', 'BarleyGreen', 'Cleaved', 'Skinned'])

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True)
plt.savefig(f"../plots/classifier-plots/ANNCONFUSION.png")
plt.show()




"""'
data = {'value': [], 'feature': [], 'grain': []}
df = pd.DataFrame.from_dict(data)
Features = np.array([f'[{i+1}]' for i in range(30)])
for i ,(imgs, labels) in enumerate(trainload):
    imgs = imgs.to(device)
    tens = ENCO(imgs)
    imgs = tens.detach().cpu().numpy()
    for img, label in tqdm(zip(imgs, labels)):
        for value, feature in zip(img**2,Features):
            new_row = {'value': value, 'feature': feature, 'grain': label}
            df = df.append(new_row, ignore_index=True)
    if i == 10:
        break
print(df[30:45])


print(df.head())

#ax = sns.pointplot(x="feature", y="value", hue="grain",data=df, palette="Set2")



sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.catplot(x='feature', y='value', row='grain', hue='grain', data=df, kind='bar', dodge=False, saturation=.5,
                ci=None, aspect=0.9)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
g.fig.set_figheight(9)
g.fig.set_figwidth(5)

# def axla(sex):
#     plt.gca().text(-.02, .2, "sex", fontweight="bold",
#             ha="right", va="center", transform=plt.gca().transAxes)
# g.map(axla,"sex")


g.fig.subplots_adjust(hspace=0.2)
# g.set_titles(" ")

g.set(yticks=[])
g.despine(bottom=True, left=True)
plt.show()
"""