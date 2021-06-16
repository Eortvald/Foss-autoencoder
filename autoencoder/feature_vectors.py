import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from datetime import *
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from CAE_model import *
from data.dataload_collection import *

X = torch.ones([1, 8,200,89]).to(device)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
aemodel = CAE(z_dim=30).to(device)
aemodel.load_state_dict(torch.load('./autoencoder/model_dicts/CAE_10Kmodel.pth', map_location=device))
aemodel.eval()

ENCO = lambda img : aemodel.encode(img)
print(ENCO(X))