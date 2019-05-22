import nibabel as nib
import numpy as np
import nilearn
# import pywt
# import matplotlib.pyplot as plt
# from scipy.misc import imresize
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
# from scipy.ndimage import zoom

# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# params = paramFunc.Run(UserInfo.__dict__, terminal=True)

dir = '/array/ssd/msmajdi/experiments/keras/exp4/test/Main/vimp2_case17_CSFn/'
im = nib.load(dir + 'PProcessed.nii.gz')
# im2 = nib.load(dir + 'reverece_contrast_PProcessed.nii.gz')


mx = im.get_data().max()
imm = 1 - im.get_data()/mx

im2 = nib.Nifti1Image(imm, im.affine)
nib.save(im2, dir+'reverece_contrast_PProcessed.nii.gz')
