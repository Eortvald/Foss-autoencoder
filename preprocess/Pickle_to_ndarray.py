import numpy as np
import pickle
import os
import datetime

def pickle_to_ndarray(root_path, save_path, n_files = 120000):
    #Make counter
    n = 0

    #List pickles in 2018/2017 samples
    pickles = os.listdir(path)
    p_len = len(pickles)

    #Create sub-folders so theres 10k in each.
    subs = []
    s = -1
    for i in range(n_files//10000):
        sub = os.path.join(save_path + 'a' + str(i))
        sub.append(subs)
        os.makedirs(sub)

    #Loop that continues until n_files (def = 120.000) is added
    for file in pickles:
        if n == n_files:
            break

        infile = open(file, 'rb')
        pic = pickle.load(infile)
        infile.close()

        for j, image in enumerate(pic):

            print(f'Pickle loaded: [{i}/{p_len}]  ------  {str(datetime.datetime.now())[11:-7]}')

            if n == n_files:
                break

            k_len = (int(float(image['attributes']['Length'])))
            k_wid = (int(float(image['attributes']['Width'])))

            if (k_len <= 190) and (k_wid <= 80):
                img = image['image']


                #Saving the image
                if n % 10000 == 0:
                    s += 1
                    print(subs[s], 'is full!')

                np.save(save_path + subs[s] + '/grain' + str(n))
                n += 1


path = "M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/validation/Images/2018 samples/"
save_path = "C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/"
pickle_to_ndarray(path, save_path)