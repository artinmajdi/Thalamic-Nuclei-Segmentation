import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from otherFuncs import smallFuncs


def myshow(ind, *argv):
    fig, axs = plt.subplots(nrows=1,ncols=len(argv))
    for ix, arg in enumerate(argv):
        axs[ix].imshow(arg[...,ind],cmap='gray')

    plt.show()


dir = '/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/test/vimp2_ctrl_911_07082013_TTO/'
im = nib.load(dir + 'WMnMPRAGE_bias_corr.nii.gz').get_data()

_, _, Allnames = smallFuncs.NucleiSelection(ind=8,organ='THALAMUS')

msk10 = nib.load(dir + 'Label/' + Allnames[8] + '.nii.gz').get_data()
msk8 = nib.load(dir + 'Label/' + Allnames[6] + '.nii.gz').get_data()
msk11 = nib.load(dir + 'Label/' + Allnames[9] + '.nii.gz').get_data()

postriorMask = 4*msk10 + msk8 + 2*msk11

myshow(137,im, msk8,msk10,msk11,postriorMask)
