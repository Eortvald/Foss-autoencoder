import numpy as np
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from datetime import *
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from autoencoder.CAE_model import CAE
import torch.onnx

batch_size = 4

input_image = torch.randn((batch_size,8,200,89))

torch_model = CAE(z_dim=30)
torch_model.load_state_dict(torch.load('model_dicts/CAE_10Kmodel.pth', map_location=None))
torch_model.eval()

x = torch.randn(1, 8, 200, 89, requires_grad=True)
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "CAE.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})




dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'relative/path/to/file/you/want')

print(dirname)
arr = np.random.randint(2, size=10)

#np.save('./data/test', arr)
