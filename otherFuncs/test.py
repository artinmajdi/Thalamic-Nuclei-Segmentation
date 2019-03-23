import numpy as np
import nibabel as nib
import os, sys
import skimage
import csv
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append( os.path.dirname(os.path.dirname(__file__)) )
# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# params = paramFunc.Run(UserInfo.__dict__)
import pandas as pd
import keras
import skimage
import nilearn

dir = '/home/artinl/Documents/research/dataset/7T/'
subject = 'vimp2_943_07242013_PA_MS/'
# im = nilearn.image.load_img(dir + subject + 'WMnMPRAGE_bias_corr.nii.gz')
# msk = nilearn.image.load_img(dir + subject + 'vtk_rois/1-THALAMUS.nii.gz')
cropMask = nilearn.image.load_img(dir + subject + 'temp/CropMask.nii.gz')

subject = 'vimp2_972_08152013_DC_MS/'
cropMask2 = nilearn.image.load_img(dir + subject + 'temp/CropMask.nii.gz')

# ! (1)
# def func_upsample(fileAddress, scale):
#     im = nilearn.image.load_img(fileAddress)
#     affine = im.affine.copy()
#     for i in range(3): affine[i,i] /= scale
#     im_US = nilearn.image.resample_img(img=im , target_affine=affine,interpolation='nearest')
#     nib.save(im_US, fileAddress.split('.nii,gz')[0] + '_US.nii.gz')
#     return im_US

# im_US = func_upsample(dir + 'PProcessed.nii.gz', 2)
# msk_US = func_upsample(dir + 'vtk_rois/1-THALAMUS_PProcessed.nii.gz', 2)

# imCrop = nilearn.image.threshold_img(img=im, mask_img=msk)

# ! (2)
# mskCroped = nilearn.image.crop_img(msk)
# imCropped = nilearn.image.resample_to_img(im,mskCroped)
# nib.save(imCropped,dir + 'PProcessed_Cropped.nii.gz')
# nib.save(mskCroped,dir + 'vtk_rois/1-THALAMUS_PProcessed_Cropped.nii.gz')

# ! (3)
# subjectList = [ s for s in os.listdir(dir) if 'vimp' in s]
# imList = [ nilearn.image.load_img(dir + subject + '/PProcessed.nii.gz') for subject in subjectList ]
# imF = nilearn.image.concat_imgs(imList, auto_resample=True)

# obj = skimage.measure.regionprops(skimage.measure.label(cropMask.get_data()))
cropMaskN = nilearn.image.resample_img(cropMask2, target_affine=cropMask.affine)

subject = 'vimp2_943_07242013_PA_MS/'
nib.save(cropMaskN,dir + subject + 'temp/CropMaskB.nii.gz')
print('---')

