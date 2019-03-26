import numpy as np
# import nibabel as nib
import os, sys
# import skimage
# import csv
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append( os.path.dirname(os.path.dirname(__file__)) )
# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# params = paramFunc.Run(UserInfo.__dict__)
# import pandas as pd
# import keras
# import skimage
# from nilearn import image as imageNilearn
import otherFuncs.smallFuncs as smallFuncs

a = smallFuncs.NucleiIndex(1,'Cascade')
FullIndex = np.append([1] , a.child)

b = smallFuncs.NucleiIndex(1,'HCascade')
list_HC = []
for ix in b.child:
    c = smallFuncs.NucleiIndex(ix,'HCascade')
    if c.child and bool(set(FullIndex) & set(c.child)): list_HC.append(ix)

a = True
b = False
c = True

if any(a,b,c): print('yes')
else: print('no')
a = np.zeros((7,3))
np.insert(a,obj=[0,0] , values=np.array([6]),axis=0)
np.insert

print('----')
# # ! (2)
# # mskCroped = imageNilearn.crop_img(msk)
# # imCropped = imageNilearn.resample_to_img(im,mskCroped)
# # nib.save(imCropped,dir + 'PProcessed_Cropped.nii.gz')
# # nib.save(mskCroped,dir + 'vtk_rois/1-THALAMUS_PProcessed_Cropped.nii.gz')

# # ! (3)
# # subjectList = [ s for s in os.listdir(dir) if 'vimp' in s]
# # imList = [ imageNilearn.load_img(dir + subject + '/PProcessed.nii.gz') for subject in subjectList ]
# # imF = imageNilearn.concat_imgs(imList, auto_resample=True)

# # obj = skimage.measure.regionprops(skimage.measure.label(cropMask.get_data()))
# cropMaskN = imageNilearn.resample_img(cropMask2, target_affine=cropMask.affine)

# subject = 'vimp2_943_07242013_PA_MS/'
# nib.save(cropMaskN,dir + subject + 'temp/CropMaskB.nii.gz')
# print('---')
