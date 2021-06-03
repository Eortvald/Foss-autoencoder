import pandas as pd
import matplotlib.pyplot as plt
labels = pd.read_csv('labels_folders_color.csv', dtype = 'unicode')

ccount = labels['ccolor'].value_counts()

plt.pie(ccount, explode = (0.1,0.1,0.1), labels = ('Green', 'Red', 'Blue'), autopct = '%1.1f%%', colors = ('#31ba79', '#fc806b','#55a6ff'))
plt.show()

