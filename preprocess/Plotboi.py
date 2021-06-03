import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
cl = pd.read_csv('Classifier_labels.csv', index_col = False)
for i in range(len(cl)):
    print(cl.loc[i][2:9])
    fc, tc = cl.loc[i][2:9][cl.loc[i][2:9] == False].count(), cl.loc[i][2:9][cl.loc[i][2:9] == True].count()
    print(tc)
    if tc == 1:
        print(cl.loc[i])
        break
