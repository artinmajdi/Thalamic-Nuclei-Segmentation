import numpy as np
np.savetxt('/array/ssd/msmajdi/a.txt', [1,0.9875938457694386596346],fmt='%1.4f')

np.loadtxt('/array/ssd/msmajdi/a.txt')

import nibabel as nib
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import otherFuncs.smallFuncs as smallFuncs
import pickle

import keras
keras.backend.binary_crossentropy

params , User = {}, {}
for i in range(1,5):
    UserInfoB = UserInfo.__dict__.copy()
    UserInfoB['DatasetIx'] = i
    User[i] = UserInfoB
    params[i] = paramFunc.Run(UserInfoB)
    print( params[i].WhichExperiment.Dataset.name )

print('---')
