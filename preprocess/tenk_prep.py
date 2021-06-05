import numpy as np
import pandas as pd
import os

#Choosing width and height cut-off
max_w = 85
max_h = 200

#Path to root of images
path = 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/BlobArchive_v2/'
savepath = ''

#Labels
df = pd.read_csv('Classifier_labels.csv')

#Looping through the 10000 first images
for i in range(1):
    sti = path + str(df.iloc[i]['folder']) + '/' + str(df.iloc[i]['Names'])
    img = np.load(sti)

    #Get width and height
    hei = np.shape(img[:, :, 0])[0]
    wid = np.shape(img[:, :, 0])[1]

    #Check if image isn't oversized
    if (hei > 200) or (wid > 85)
        print(df.iloc[i]['ccolor'])

    else:
        #Apply mask
        mask = img[:,:,7]
        img = np.where(mask[...,None] != 0, img, [0,0,0,0,0,0,0,0])

        #Calculate the number of pixels to be padded
        if (hei % 2) == 0:
            rhei1 = (max_h - hei) / 2
            rhei2 = (max_h - hei) / 2
        elif (hei % 2) == 1:
            rhei1 = (max_h - hei + 1) / 2
            rhei2 = (max_h - hei - 1) / 2
        if (wid % 2) == 1:
            rwid1 = (max_w - wid) / 2
            rwid2 = (max_w - wid) / 2
        elif (wid % 2) == 0:
            rwid1 = (max_w - wid + 1) / 2
            rwid2 = (max_w - wid - 1) / 2

        # Zero padding
        img = np.pad(img, ((int(rhei2),int(rhei1)), (int(rwid1),int(rwid2)), (0,0)), 'constant')
        print(np.shape(img))