import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')
import matplotlib.pyplot as plt
import numpy as np
from Parameters import UserInfo, paramFunc
import nibabel as nib
from scipy import ndimage
from preprocess import croppingA
params = paramFunc.Run(UserInfo.__dict__)

params.directories.Test.Result.split('/subExp')[0]
a = np.random.randint(0,9,size=(4,3,2))
a[...,0]
np.savetxt('/array/ssd/msmajdi/code/thalamus/keras/out.txt',a[...,0],fmt='%d')

names = ['aa','bb','cc','dd']
np.savetxt('/array/ssd/msmajdi/code/thalamus/keras/names.txt',names,fmt='%s')

e = np.loadtxt('/array/ssd/msmajdi/code/thalamus/keras/out.txt',dtype=int)
names2 = np.loadtxt('/array/ssd/msmajdi/code/thalamus/keras/names.txt',dtype=str)

e
names2[1]









































print('----')
