import nibabel as nib
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.misc import imresize
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
from scipy.ndimage import zoom

# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# params = paramFunc.Run(UserInfo.__dict__, terminal=True)

dir = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/pre-steps/WMn/step1_uncropped/case1/'

im = nib.load(dir + 'LIFUP001_MPRAGE_WMn.nii.gz')
im2 = zoom(im.get_data() , 1/0.7)
smallFuncs.saveImage(im2,im.affine,im.header,dir+'junk.nii.gz')


im = nib.load(dir + 'Label/1-THALAMUS.nii.gz')
im2 = zoom(im.get_data() , 1/0.7)
smallFuncs.saveImage(im2,im.affine,im.header,dir+'Label/1-THALAMUS_junk.nii.gz')



