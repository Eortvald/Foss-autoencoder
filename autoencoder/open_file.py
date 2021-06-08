import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import *

path = 'model_dicts/'

train = np.load(path+'train_results.npy', allow_pickle=True)
test = np.load(path+'test_results.npy', allow_pickle=True)

print(len(train),len(test))

plt.plot(np.arange(len(train)), train, label='train')  # etc.
#plt.plot(np.arange(len(test)), test, label='test')
plt.xlabel('acummulated batches')
plt.ylabel('Loss')
plt.title("Train vs Test")
plt.legend()

plt.savefig(f"plots/autoencoder-plots/test_{str(datetime.now())[5:-10].replace(' ','_').replace(':','-')}.png")
#plt.show()

