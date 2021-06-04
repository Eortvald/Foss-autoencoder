import pandas as pd

labels = pd.read_csv('labels_folders_color.csv', low_memory=False)
id = list()
for i in range(len(labels)):
    fc, tc = labels.loc[i][1:8][labels.loc[i][1:8] == False].count(), labels.loc[i][1:8][labels.loc[i][1:8] == True].count()
    if tc == 0:
        id.append(i)
cl = labels.drop(id)
cl.to_csv('Classifier_labels.csv', index = False)