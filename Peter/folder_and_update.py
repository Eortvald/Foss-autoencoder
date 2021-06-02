import pandas as pd

df = pd.read_csv('labels.csv')
df_folder = pd.read_csv('labels_folder.csv')
df_updated = pd.read_csv('labels_color.csv')

df.insert(loc = len(df.columns), column = 'folder', value = df_folder['folder'])
df.insert(loc = len(df.columns), column = 'ccolor', value = df_updated['ccolor'])

df.to_csv('labels_folders_color.csv', index = False)