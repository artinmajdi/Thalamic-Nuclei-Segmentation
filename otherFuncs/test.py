import nibabel as nib
import numpy as np
import nilearn
import matplotlib.pyplot as plt
import os, sys
from skimage import measure
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs

def imShow(*args):
    _, axes = plt.subplots(1,len(args))
    for ax, im in enumerate(args):
        axes[ax].imshow(im,cmap='gray')

    plt.show()

    return True

dir = '/array/ssd/msmajdi/data/originals/CSFn/Dataset2_with_Manual_Labels/ET/CSFn/F/'

im = nib.load(dir + '7T_pre_WMnMPRAGE.nii.gz').get_data()
imc = nib.load(dir + 'csfn.nii.gz').get_data()
msk = nib.load(dir + 'truth/1_THALAMUS.nii.gz').get_data()

im2  = im/im.max()   + msk*0.5
# imc2 = imc/imc.max() + msk*0.5
plt.imshow(np.squeeze(im2[:,200,:]),cmap='gray')
imShow()

