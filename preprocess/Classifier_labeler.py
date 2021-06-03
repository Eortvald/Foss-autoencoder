import pandas as pd

labels = pd.read_csv('labels_folders_color.csv', low_memory=False)
id = list()
for i in range(len(labels)):
    fc, tc = labels.loc[i][1:7][labels.loc[i][1:7] == False].count(), labels.loc[i][1:7][labels.loc[i][1:7] == True].count()
    if (fc < 6) or (tc < 0):
        id.append(i)
    else:
        print(i)
cl = labels.drop(id)
cl.to_csv('Classifier_labels.csv')