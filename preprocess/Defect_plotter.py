import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import random

def defect_plotter(root_path, labels_path):
    # Later used for plotting
    fig, ar = plt.subplots(1, 7, sharey = True, sharex = True, constrained_layout = True)

    # Opening labels
    df = pd.read_csv(labels_path)

    #Initializing defects
    defect_list = ['Oat', 'Broken', 'Rye', 'Skinned', 'Cleaved', 'Wheat', 'BarleyGreen']

    for i, defects in enumerate(defect_list):
        options = df.loc[df[defects] == True]
        choice = options.iloc[random.randint(0, len(options))]
        img = np.load(root_path + choice['folder'] + '/' + choice['Names'] + '.npy')
        ar[i].imshow(np.dstack((img[:,:,4], img[:,:,2], img[:,:,1])))
        ar[i].set_title(defects)
        ar[i].set_xlim(0,np.shape(img)[1])
        ar[i].set_ylim(0,np.shape(img)[0])
        if defects == 'BarleyGreen':
            ar[i].set_title('Green')
    plt.savefig('defects')
    plt.show()
defect_plotter('M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/', 'labels_folders_color.csv')