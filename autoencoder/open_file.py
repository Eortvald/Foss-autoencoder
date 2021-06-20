import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from datetime import *

labels = ['Oat', 'Broken', 'Rye', 'Wheat', 'BarleyGreen', 'Cleaved', 'Skinned', 'Barley']
arr0 = 2.5 * np.random.randn(30) + 10
arr1 = 2.5 * np.random.randn(30) + 12
arr2 = 2.5 * np.random.randn(30) + 13
arr3 = 2.5 * np.random.randn(30) + 14
arr4 = 2.5 * np.random.randn(30) + 18
arr5 = 2.5 * np.random.randn(30) + 8
arr6 = 2.5 * np.random.randn(30) + 9
arr7 = 2.5 * np.random.randn(30) + 11

MA = np.random.rand(8,30)

Features = np.array([f'[{i+1}]' for i in range(30)])

data = {'value': [], 'feature': [], 'grain': []}
df = pd.DataFrame.from_dict(data)


for i, grain_vec in enumerate(MA):
    for v,f in zip(grain_vec, Features):
        new_row = {'value': v, 'feature': f, 'grain': f'grain_{i}'}
        df = df.append(new_row, ignore_index=True)

print(df)