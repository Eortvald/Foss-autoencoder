import pandas as pd

labels = pd.read_csv('labels.csv')
trues = labels.sum()
print(trues)
