import nibabel as nib
import numpy as np
import nilearn
import matplotlib.pyplot as plt
import os, sys
from skimage import measure
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs


dir = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/pre-steps/CSFn/full_Image/step1_registered_labels/vimp2_case17/'

im = nib.load(dir + 'full_T1.nii.gz')
msk = nib.load(dir + 'Label/1-THALAMUS.nii.gz')

obj = measure.regionprops(measure.label(msk.get_data()))

sc = int((obj[0].bbox[-1] + obj[0].bbox[2])/2)

imm = np.squeeze(im.slicer[:,:,sc:sc+1].get_data())
mskk = np.squeeze(msk.slicer[:,:,sc:sc+1].get_data())

imm2 = imm/imm.max() + mskk*0.5
smallFuncs.imShow(imm2 , imm)
# plt.imshow(imm2,cmap='gray')
