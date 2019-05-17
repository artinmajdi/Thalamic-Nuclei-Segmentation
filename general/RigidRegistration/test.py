import nibabel as nib
import numpy as np
import os



def imShow(*args):
    _, axes = plt.subplots(len(args))
    for ax, im in enumerate(args):
        axes[ax].imshow(im[...,ax],cmap='gray')

    plt.show()

    return True

os.getcwd()
dir = '/array/ssd/msmajdi/code/general/RigidRegistration/'

im = nib.load(dir + 'origtemplate.nii.gz').get_data()
msk = nib.load(dir + 'MyCrop_Template2_Gap20.nii.gz').get_data()

cropMask
