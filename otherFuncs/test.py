import nibabel as nib
import numpy as np
import nilearn
import matplotlib.pyplot as plt
import os, sys
from skimage import measure
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
params = paramFunc.Run(UserInfo.__dict__, terminal=False)






