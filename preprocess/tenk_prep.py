import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Choosing width and height cut-off
max_w = 224
max_h = 224
n = 0
i = 0
#Path to root of images
path = 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/BlobArchive_v2/'
savepath = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs224/'

#Labels
df = pd.read_csv('Classifier_labels.csv')

#Looping through the 10000 first images
while n < 10000:
    i += 1
    sti = path + str(df.iloc[i]['folder']) + '/' + str(df.iloc[i]['Names'] + '.npy')
    img = np.load(sti)

    #Get width and height
    hei = np.shape(img[:, :, 0])[0]
    wid = np.shape(img[:, :, 0])[1]

    #Check if image isn't oversized
    if (hei <= max_h) and (wid <= max_w):
        n += 1
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
        if (wid % 2) == 0:
            rwid1 = (max_w - wid) / 2
            rwid2 = (max_w - wid) / 2
        elif (wid % 2) == 1:
            rwid1 = (max_w - wid + 1) / 2
            rwid2 = (max_w - wid - 1) / 2

        # Zero padding
        img = np.pad(img, ((int(rhei2),int(rhei1)), (int(rwid1),int(rwid2)), (0,0)), 'constant')

        #Saving edited image.
        if n % 5 == 0:
            folderpath = 'test/'
        else:
            folderpath = 'train/'
        savename = savepath + folderpath + str(df.iloc[i]['Names'])

        #Get the label as string (e.g. 'Wheat' or 'Cleaved')
        label = str(df.loc[i][1:8][df.loc[i][1:8] == True]).split(' ')[0]

        #New array that contains image and label
        img_labeled = np.array([img.astype("float32"), label], dtype = object)
        print(savename)
        #Save the image + label
        np.save(savename, img_labeled)