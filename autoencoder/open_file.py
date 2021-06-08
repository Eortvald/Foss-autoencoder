import numpy as np
import pickle
import os

path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Validation/Images/2017 samples/'

pickle_me = open(path+'110-E001_1.pkl', "rb")

file = pickle.load(pickle_me)

arr = np.array(file)

print(arr[0])