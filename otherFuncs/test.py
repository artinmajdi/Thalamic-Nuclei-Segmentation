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

dir = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/pre-steps/CSFn/cropped_Image/step1_registered_labels/vimp2_case2/Label/'
im = nib.load(dir + '1-THALAMUS.nii.gz')

im2 = nib.Nifti1Image(im.get_data() > 0.5 , im.affine)
nib.save(im2, dir+'1t_junk.nii.gz')
