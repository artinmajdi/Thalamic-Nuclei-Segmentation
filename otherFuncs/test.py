import numpy as np
import os
from imageio import imread
import matplotlib.pyplot as plt
import cv2
dir = '/array/ssd/msmajdi/data/KaggleCompetition/train'
subF = next(os.walk(dir))

for ind in range(len(subF[1])):

    imDir = subF[0] + '/' + subF[1][ind] + '/images'
    imMsk = subF[0] + '/' + subF[1][ind] + '/masks'
    a = next(os.walk(imMsk))
    b = next(os.walk(imDir))

    im = np.expand_dims(imread(imDir + '/' + b[2][0])[...,:3],axis=0)
    print(im.shape)
    # images = im if ind == 0 else np.concatenate((images,im),axis=0)
    #
    # msk = imread(imMsk + '/' + a[2][0]) if len(a[2]) > 0 else np.zeros(im.shape)
    # if len(a[2]) > 1:
    #     for sF in range(1,len(a[2])):
    #         msk = msk + imread(imMsk + '/' + a[2][sF])
    #
    # msk = np.expand_dims(msk,axis=0)
    # masks = msk if ind == 0 else np.concatenate((masks,msk),axis=0)


# fig , ax = plt.subplots(1,2)
# ax[0].imshow(images[1,...],cmap='gray')
# ax[1].imshow(masks[1,...],cmap='gray')
# plt.show()
