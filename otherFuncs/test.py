import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Parameters import UserInfo, paramFunc
from otherFuncs import smallFuncs
from scipy import ndimage


UserInfoB = UserInfo.__dict__

def readImageMask(UserInfoB,sjIx):
    params = paramFunc.Run(UserInfoB)
    subjects = params.directories.Test.Input.Subjects
    subject = subjects[list(subjects)[sjIx]]
    print(list(subjects)[sjIx])

    im = nib.load(subject.address + '/' + subject.ImageOriginal + '.nii.gz').get_data()
    msk = nib.load(subject.Label.address + '/' + subject.Label.LabelOriginal + '.nii.gz').get_data()
    pred = nib.load(params.directories.Test.Result + '/' + list(subjects)[sjIx] + '/' + subject.Label.LabelOriginal + '.nii.gz').get_data()

    return im, msk, pred

dir = '/array/ssd/msmajdi/data/originals/ET/prunnedForReSlicing/7T/vimp2_B'
im = nib.load(dir + '/WMnMPRAGE_bias_corr.nii.gz').get_data()
msk = nib.load(dir + '/Label/1-THALAMUS.nii.gz').get_data()
pred = nib.load(dir + '/Label/1-THALAMUS2.nii.gz').get_data()

UserInfoB['nucleus_Index'] = [1]
im, msk, pred = readImageMask(UserInfoB,8)
a = nib.viewers.OrthoSlicer3D(msk,title='manual')
b = nib.viewers.OrthoSlicer3D(pred,title='prediction')
c = nib.viewers.OrthoSlicer3D(im,title='image')
c.link_to(a)
c.link_to(b)
c.show()
print('---')
# c.show()
# plt.show()
