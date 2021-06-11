import numpy as np
import os
import matplotlib.pyplot as plt

def _plot_transformation(img, max_h = 180, max_w = 80):
        pre = img.copy()

        mask = img[:, :, 7]
        img = np.where(mask[..., None] != 0, img, [0, 0, 0, 0, 0, 0, 0, 0])

        # width and height of image
        h = np.shape(img[:, :, 0])[0]
        w = np.shape(img[:, :, 0])[1]
        print(w)
        print(np.shape(img))

        pre_crop = img.copy()

        # Trim/Crop if image is too large
        if w > max_w:
            img = np.delete(img, np.where(np.sum(mask, axis=0) == 0)[0], axis=1)
            w = np.shape(img[:, :, 0])[1]
            print(w)
            print(np.shape(img))

        if h > max_h:
            img = np.delete(img, np.where(np.sum(mask, axis=1) == 0)[0], axis=0)
            h = np.shape(img[:, :, 7])[0]
        pre_pad = img.copy()

        if (h % 2) == 0:
            rh1 = (max_h - h) / 2
            rh2 = (max_h - h) / 2
        elif (h % 2) == 1:
            rh1 = (max_h - h + 1) / 2
            rh2 = (max_h - h - 1) / 2
        if (w % 2) == 0:
            rw1 = (max_w - w) / 2
            rw2 = (max_w - w) / 2
        elif (w % 2) == 1:
            rw1 = (max_w - w + 1) / 2
            rw2 = (max_w - w - 1) / 2

        # Zero padding
        post_pad = np.pad(img, ((int(rh2), int(rh1)), (int(rw1), int(rw2)), (0, 0)), 'constant')

        fig, ar = plt.subplots(1, 4, sharex=True, sharey=True)
        ar[0].imshow(np.dstack((pre[:,:,4], pre[:,:,2], pre[:,:,1])))
        ar[1].imshow(np.dstack((pre_crop[:,:,4], pre_crop[:,:,2], pre_crop[:,:,1])))
        ar[2].imshow(np.dstack((pre_pad[:,:,4], pre_pad[:,:,2], pre_pad[:,:,1])))
        ar[3].imshow(np.dstack((post_pad[:,:,4], post_pad[:,:,2], post_pad[:,:,1])))
        plt.savefig('C:/Users/Nullerh/Documents/DTU_SCHOOL_WORK/Semester4/02466_Project/KORN/plots/preprocess-plots/pre'
                    + img_name.split(".")[0])
        plt.show()

img_path = 'C:/Users/Nullerh/Desktop/temp/'
img_name = '5c2f4c564162bb128cfb1900.npy'
img = np.load(img_path + img_name)
_plot_transformation(img, max_w = 50, max_h = 160)

