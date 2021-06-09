import numpy as np
import os
import pickle

path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/validation/Images/'

with os.scandir(path + '2017 samples/') as entries:
    for entry in entries:
        infile = open(entry, 'rb')
        pic = pickle.load(infile)
        infile.close()
        print(pic, len(pic))
        break
with os.scandir(path + '2018 samples/') as entries:
    for entry in entries:
        infile = open(entry, 'rb')
        pic = pickle.load(infile)
        infile.close()
        print(pic, len(pic))
        break