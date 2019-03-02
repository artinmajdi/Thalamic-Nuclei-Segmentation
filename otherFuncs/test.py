import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

def myshow(im,msk,ind):
    fig, axs = plt.subplots(nrows=1,ncols=2)
    axs[0].imshow(im[...,ind],cmap='gray')
    axs[1].imshow(msk[...,ind],cmap='gray')
    plt.show()


dir = '/array/ssd/msmajdi/data/preProcessed/7T/DividedByDisease/ET/vimp2_C_ET7T/'
im = nib.load(dir + 'WMnMPRAGE_bias_corr.nii.gz').get_data()
msk = nib.load(dir + 'Label/1-THALAMUS.nii.gz').get_data()

myshow(im,msk,ind=120)
