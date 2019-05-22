import nibabel as nib
import numpy as np
import nilearn
# import pywt
import pybm3d

import matplotlib.pyplot as plt
# from scipy.misc import imresize
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
# from scipy.ndimage import zoom

# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# params = paramFunc.Run(UserInfo.__dict__, terminal=True)

dir = '/array/ssd/msmajdi/experiments/keras/Check_Datasets/CSFn_Checks/'
subject = 'vimp2_case17_CSFn'

dir2 = dir + subject + '/PProcessed.nii.gz'
im = nib.load(dir2)

mx = im.get_data().max()
imm = 1 - im.get_data()/mx


imm2 = pybm3d.bm3d.bm3d(imm,np.std(imm))


# plt.imshow(imm[...,80],cmap='gray')
# plt.imshow(imm2[...,80],cmap='gray')
a = nib.viewers.OrthoSlicer3D(imm)
b = nib.viewers.OrthoSlicer3D(imm2,title='denoised')
a.link_to(b)
a.show()
# im2 = nib.Nifti1Image(imm, im.affine)
# nib.save(im2, dir+'reverece_contrast_PProcessed.nii.gz')
