import numpy as np
import nibabel as nib
# import matplotlib.pyplot as plt
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Parameters import UserInfo, paramFunc
from otherFuncs import smallFuncs
# from scipy import ndimage
# from nilearn import image
# from skimage import feature
from skimage import measure

param = paramFunc.Run(UserInfo.__dict__)
subejcts = param.directories.Test.Input.Subjects

label,_ , _ = smallFuncs.NucleiSelection(ind=1,organ='THALAMUS')
im = nib.load('/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/test/vimp2_K/PProcessed.nii.gz').get_data()
msk = nib.load('/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/test/vimp2_K/Label/' + label + '_PProcessed.nii.gz').get_data()
pred = nib.load('/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/results/subExp2_MinMax_wAug_Loss_BCE_nl3/vimp2_L/' + label + '.nii.gz').get_data()


# bb = measure.regionprops(msk.astype(np.int32))[0].bbox
#
a = nib.viewers.OrthoSlicer3D(im,title='image')
b = nib.viewers.OrthoSlicer3D(msk,title='Label')
c = nib.viewers.OrthoSlicer3D(pred,title='pred')
b.link_to(a)
b.link_to(c)
b.show()
