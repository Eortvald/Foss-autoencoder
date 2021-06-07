import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.load('height.npy')
y = np.load('widths.npy')

df = pd.read_csv('labels_folders_color.csv')
print(df.loc[df['Names'] == '5ebe7e0de2c404484c770dff'])
s_df = pd.DataFrame()
for folder in sorted(df['folder'].unique()):
    s_df = s_df.append((df.loc[df['folder'] == folder].sort_values('Names')))

colors = s_df['ccolor'].tolist()
print(len(x), len(y), len(colors))
dic = {'Green': 'g', 'Blue': 'b', 'Red': 'r'}

img = np.load('M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/ff/5ebe7e0de2c404484c770dff.npy')
plt.imshow(np.dstack((img[:,:,4], img[:,:,2], img[:,:,1])))
plt.show()
c = [dic.get(n,n) for n in colors]
'''
print(len(x), len(y), len(c))
plt.scatter(x = x, y = y, c = c)
plt.show()
'''