import os
import sys
# sys.path.append(os.path.dirname(__file__))  # 
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
from modelFuncs.Metrics import Precision_Recall_Curve
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

dir = '/array/ssd/msmajdi/experiments/keras/exp6/results/sE13_Cascade_FM20_ResFCN_ResUnet2_TL_NL3_LS_MyLogDice_US1_FCNA0_FCNB0_FM0_permute0_CSFn2_TL_Main_CV_a_for_percision_recall_curve/sd2/vimp2_A_CSFn2/'
dirM = '/array/ssd/msmajdi/experiments/keras/exp6/crossVal/CSFn2/a/vimp2_A_CSFn2/Label/'
a = smallFuncs.Nuclei_Class(index=1,method='Cascade').All_Nuclei()

msk = nib.load(dir + a.Names[0] + '.nii.gz').get_data()
mskM = nib.load(dirM + a.Names[0] + '.nii.gz').get_data()


plt.plot(np.unique(msk))
Precision_Recall_Curve(mskM,msk)

print('---')
