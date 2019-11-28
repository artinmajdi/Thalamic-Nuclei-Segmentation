import os
import sys
# sys.path.append(os.path.dirname(__file__))  # 
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# import otherFuncs.smallFuncs as smallFuncs
from modelFuncs.Metrics import Precision_Recall_Curve
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

dir = '/array/ssd/msmajdi/experiments/keras/exp7/results/sE12_Cascade_FM00_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/2.5D_MV/vimp2_988_wmn/'
dirM = '/array/ssd/msmajdi/experiments/keras/exp7/crossVal/temp/Main/b/vimp2_988_08302013_CB/Label/'
# a = smallFuncs.Nuclei_Class(index=1,method='Cascade').All_Nuclei()

msk = nib.load(dir  + '8-Pul.nii.gz').get_data()
mskM = nib.load(dirM  + '8-Pul_PProcessed.nii.gz').get_data()


plt.plot(np.unique(msk))
# plt.show()
Precision_Recall_Curve(mskM,msk)

yp1 = np.reshape(mskM,[-1,1])
yt1 = np.reshape(msk,[-1,1])

precision, recall, thresholds = precision_recall_curve(yt1, yp1)
average_precision = average_precision_score(yt1, yp1)

print('---')
