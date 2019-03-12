import numpy as np

K = []
for i in range(1000):
    K.append(np.random.randint(low=0, high=2))

np.unique(K)

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


#
a = nib.viewers.OrthoSlicer3D(im,title='image')

b.show()
