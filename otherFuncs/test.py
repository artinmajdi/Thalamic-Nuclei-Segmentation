import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')
import matplotlib.pyplot as plt
import numpy as np
from Parameters import UserInfo, paramFunc
import nibabel as nib
from scipy import ndimage
from preprocess import croppingA
params = paramFunc.Run(UserInfo.__dict__)
import numpy as np
a = [tuple([1,2]),  tuple([8,4]),tuple([ -2, 9])]
a = [ [1,2] ,  [8,4] , [ -2, 9]]

np.savetxt('/array/ssd/msmajdi/code/thalamus/keras/out.txt',a,fmt='%d')

np.savetxt('/array/ssd/msmajdi/code/thalamus/keras/names.txt',names,fmt='%s')

e = np.loadtxt('/array/ssd/msmajdi/code/thalamus/keras/out.txt',dtype=int)
names2 = np.loadtxt('/array/ssd/msmajdi/code/thalamus/keras/names.txt',dtype=str)

e
names2[1]









































print('----')
