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


sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create the data

g = np.tile(list("ABCDEFGHIJ"), 5)
df = pd.DataFrame(dict(x=x, g=g))

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)
g.map(plt.axhline, y=0, lw=2, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "x")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)


