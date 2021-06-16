import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from torch import nn, optim
from data.dataload_collection import *
from autoencoder.CAE_model import *
import os



device = 'cuda' if torch.cuda.is_available() else 'cpu'
aemodel = CAE(z_dim=30).to(device)
aemodel.load_state_dict(torch.load('./autoencoder/model_dicts/CAE_10Kmodel.pth', map_location=device))
aemodel.eval()

AUTOENCODE = lambda img: aemodel(img)

savepath = '../plots/autoencoder-plots/reconstructions/'


for i in range(4):
    J = np.random.randint(1,1000)
    img = dataloader[J]
    np.save(os.path.join(savepath,f'Input_{J}'),img.detach().cpu().numpy())
    recon = AUTOENCODE(img)
    np.save(os.path.join(savepath,f'RECON_{J}'),recon.detach().cpu().numpy())


