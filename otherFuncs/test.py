import numpy as np
import nibabel as nib
# import matplotlib.pyplot as plt
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from Parameters import UserInfo, paramFunc
# from otherFuncs import smallFuncs
# from scipy import ndimage
# from nilearn import image
# from skimage import feature
# UserInfoB = UserInfo.__dict__


dir = '/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/test'


subject = '/vimp2_N'
im = nib.load(dir + subject + '/PProcessed.nii.gz').get_data()
msk = nib.load(dir + subject + '/Label/1-THALAMUS_PProcessed.nii.gz').get_data()

im2 = im + 1000*msk
nib.viewers.OrthoSlicer3D(im2,title='image').show()

# plt.imshow(im2[...,30],cmap='gray')
# a = nib.viewers.OrthoSlicer3D(im,title='image')
# b = nib.viewers.OrthoSlicer3D(msk,title='Label')
# b.link_to(a)
# b.show()
