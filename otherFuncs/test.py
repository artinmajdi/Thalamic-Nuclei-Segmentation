import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import numpy as np
from Parameters import UserInfo, paramFunc
from otherFuncs import datasets, smallFuncs


UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
UserInfoB['TestOnly'] = True
UserInfoB['CreatingTheExperiment'] = False

im = {}
for dx in [1,4]:
    UserInfoB['DatasetIx'] = dx
    im[dx] = {}
    for ix in range(3):
        UserInfoB['slicingDim'] = ix
        params = paramFunc.Run(UserInfoB)
        Data, params, Info = datasets.check_Dataset_ForTraining(params=params, flag=True, Info={})
        name = list(Data.Test)[0]
        im[dx]['subject'] = name
        im[dx][ix] = Data.Test[name].Image.transpose([1,2,0,3]).squeeze()

im[4]['subject']
fig , axs = plt.subplots(1,3)
axs[0].imshow(im[1][0][:,:,38].squeeze(),cmap='gray')
axs[0].set_title('3T axis:0')
axs[1].imshow(im[1][1][:,:,38].squeeze(),cmap='gray')
axs[1].set_title('axis:1')
axs[2].imshow(im[1][2][:,:,18].squeeze(),cmap='gray')
axs[2].set_title('axis:2')
plt.show()


im[4][0].shape
im[4][1].shape
im[4][2].shape


fig , axs = plt.subplots(1,3)
axs[0].imshow(im[4][0][:,:,38].squeeze(),cmap='gray')
axs[0].set_title('7T axis:0')
axs[1].imshow(im[4][1][:,:,41].squeeze(),cmap='gray')
axs[1].set_title('axis:1')
axs[2].imshow(im[4][2][:,:,40].squeeze(),cmap='gray')
axs[2].set_title('axis:2')
plt.show()
