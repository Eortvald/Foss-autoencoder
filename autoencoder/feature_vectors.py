import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch, torchvision
import pandas as pd
import seaborn as sns
from torch import nn, optim
from datetime import *
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from CAE_model import *
#from data.dataload_collection import *
import torch, torchvision
import matplotlib.pyplot as plt
from datetime import *
import autoencoder, classifier, preprocess
device = 'cuda' if torch.cuda.is_available() else 'cpu'


X = torch.ones([4, 8,180,80]).to(device)

aemodel = CAE(z_dim=30).to(device)
aemodel.load_state_dict(torch.load('model_dicts/CAE_10Kmodel.pth', map_location=device))
aemodel.eval()
ENCO = lambda img : aemodel.encode(img)
#for i, (data, label) in enumerate(dataloader):
 #   ENCO(data)


#Making data matrixes for each class
row = np.arange(30)
matrix = np.array([row for i in range(100)])

print(matrix.mean(0))

feature_class_vectors = {'Oat':1,
                         'Broken':2,
                         'Rye':3,
                         'Wheat':4,
                         'BarleyGreen':5,
                         'Cleaved':6,
                         'Skinned':7,
                         'Barley*':8}





sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

#g = np.tile(['Oat', 'Broken', 'Rye', 'Wheat', 'BarleyGreen', 'Cleaved', 'Skinned', 'Barley*'], 8)

# Create some random distribution data and labels, and store them in a dataframe
rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list("ABCDEFGHIJ"), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)
df["x"] += m

# Initialize the FacetGrid chart object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
# g = sns.FacetGrid(df, row="g", hue="g", aspect=6, height=1.0, palette=pal)
g = sns.FacetGrid(df, row="g", hue="g", palette=pal)

''' Alternatively set figsize using the following 2 parameters.'''
g.fig.set_figheight(5.5)
g.fig.set_figwidth(7)
# or use plt.gcf().set_size_inches(12, 12)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw_adjust=.2)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
# Values x,y in ax.text(x,y, ...) controls the x,y offset from the x axis.
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

# Use ``map()`` to calculate the label positions
g.map(label, "x")

# Set the subplots to overlap slightly on their vertical direction
g.fig.subplots_adjust(hspace=-0.5)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)


