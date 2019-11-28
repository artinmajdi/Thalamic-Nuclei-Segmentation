import os
import sys
# sys.path.append(os.path.dirname(__file__))  # 
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
from modelFuncs.Metrics import Precision_Recall_Curve
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

dir = '/array/ssd/msmajdi/experiments/keras/exp7/results/sE12_Cascade_FM00_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/2.5D_MV/vimp2_988_wmn/'
dirM = '/array/ssd/msmajdi/experiments/keras/exp7/crossVal/temp/Main/b/vimp2_988_08302013_CB/Label/'
a = smallFuncs.Nuclei_Class(index=1,method='Cascade').All_Nuclei()

msk = nib.load(dir  + '8-Pul.nii.gz').get_data()
mskM = nib.load(dirM  + '8-Pul_PProcessed.nii.gz').get_data()


plt.plot(np.unique(msk))
plt.show()
Precision_Recall_Curve(mskM,msk)

print('---')
