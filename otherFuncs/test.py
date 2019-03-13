# import numpy as np
# import nibabel as nib
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from Parameters import UserInfo, paramFunc
# from otherFuncs import smallFuncs
# import pickle

params , User = {}, {}
for i in range(1,5):
    UserInfoB = UserInfo.__dict__.copy()
    UserInfoB['DatasetIx'] = i
    User[i] = UserInfoB
    params[i] = paramFunc.Run(UserInfoB)
    print( params[i].WhichExperiment.Dataset.name )

print('---')
