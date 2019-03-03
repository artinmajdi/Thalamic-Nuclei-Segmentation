import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os, sys
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
from scipy import ndimage

def myshow(ind, *argv):
    fig, axs = plt.subplots(nrows=1,ncols=len(argv))
    for ix, arg in enumerate(argv):
        axs[ix].imshow(arg[...,ind],cmap='gray')

    plt.show()



# myshow(137,im, msk8,msk10,msk11,postriorMask)
