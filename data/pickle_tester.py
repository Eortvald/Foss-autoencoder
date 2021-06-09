import numpy as np
import os
from os import listdir
import pickle
import datetime
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/validation/Images/'

widths = list()
heights = list()

def pickle_dir(path: str, subfolder: str):
    path = path + subfolder

    folder = listdir(path)
    print(folder)
    folder_images = len(folder)
    for i, file in enumerate(folder):

        print(f'Pickle loaded: [{i}/{folder_images}]  ------  {str(datetime.datetime.now())[11:-7]}')
        infile = open(file, 'rb')
        pic = pickle.load(infile)
        infile.close()

        picl = len(pic)
        for j, image in enumerate(pic):

            print(f'\t --[{j}/{picl}]')
            heights.append(int(float(image['attributes']['Length'])))
            widths.append(int(float(image['attributes']['Width'])))

pickle_dir(path, '2017 samples/')

pickle_dir(path, '2018 samples/')


# with os.scandir(path + '2017 samples/') as entries:
#     for entry in entries:
#         infile = open(entry, 'rb')
#         pic = pickle.load(infile)
#         infile.close()
#         for image in pic:
#             heights.append(int(float(image['attributes']['Length'])))
#             widths.append(int(float(image['attributes']['Width'])))
#
# with os.scandir(path + '2018 samples/') as entries:
#     for entry in entries:
#         infile = open(entry, 'rb')
#         pic = pickle.load(infile)
#         infile.close()
#         for image in pic:
#             heights.append(int(float(image['attributes']['Length'])))
#             widths.append(int(float(image['attributes']['Width'])))

upper = max(heights)
plt.hist([widths, heights], bins = 70, label = ['widths', 'heights'])
plt.legend(loc='upper right')
plt.xticks((np.linspace(upper % 35, upper, upper//35)).astype(int))
plt.savefig('C:/Users/Ext1306/PycharmProjects/Foss-autoencoder/plots/preprocess-plots/sizedist')
np.save('C:/Users/Ext1306/PycharmProjects/Foss-autoencoder/preprocess/pickle_width.npy', widths)
np.save('C:/Users/Ext1306/PycharmProjects/Foss-autoencoder/preprocess/pickle_height.npy', heights)