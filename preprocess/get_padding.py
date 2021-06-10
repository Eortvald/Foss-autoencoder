import numpy as np

def get_padded_and_masked_image(img, max_h = 180, max_w = 80):
    #Apply mask
    mask = img[:, :, 7]
    img = np.where(mask[..., None] != 0, img, [0, 0, 0, 0, 0, 0, 0, 0])

    # width and height of image
    hei = np.shape(img[:, :, 0])[0]
    wid = np.shape(img[:, :, 0])[1]

    if (hei % 2) == 0:
        rhei1 = (max_h - hei) / 2
        rhei2 = (max_h - hei) / 2
    elif (hei % 2) == 1:
        rhei1 = (max_h - hei + 1) / 2
        rhei2 = (max_h - hei - 1) / 2
    if (wid % 2) == 0:
        rwid1 = (max_w - wid) / 2
        rwid2 = (max_w - wid) / 2
    elif (wid % 2) == 1:
        rwid1 = (max_w - wid + 1) / 2
        rwid2 = (max_w - wid - 1) / 2

    # Zero padding
    return np.pad(img, ((int(rhei2),int(rhei1)), (int(rwid1),int(rwid2)), (0,0)), 'constant')