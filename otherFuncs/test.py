import numpy as np
import nibabel as nib
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
params = paramFunc.Run(UserInfo.__dict__)


Dir = '/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/train/SRI/vimp2_1604_10092015/'
Dir2 = '/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/train/temp/vimp2_B/'

im = nib.load(Dir + 'PProcessed.nii.gz').get_data()
im2 = nib.load(Dir2 + 'PProcessed.nii.gz').get_data()

im.shape
im2.shape


a = nib.viewers.OrthoSlicer3D(im)
b = nib.viewers.OrthoSlicer3D(msk)
a.link_to(b)
a.show()
print('---')
