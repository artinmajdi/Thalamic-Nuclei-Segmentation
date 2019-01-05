
import os, sys
__file__ = '/array/ssd/msmajdi/code/thalamus/keras'  #! only if I'm using Hydrogen Atom
sys.path.append(__file__)
from otherFuncs import params
import numpy as np
import nibabel as nib

imF = nib.load('/array/ssd/msmajdi/experiments/keras/exp1_tmp/train/vimp2_2309_07072016/WMnMPRAGE_bias_corr.nii.gz')

header = imF.get_header()
affine = imF.get_affine()
print('---')