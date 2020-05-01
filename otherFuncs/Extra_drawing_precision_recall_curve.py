import os
import sys
# sys.path.append(os.path.dirname(__file__))  # 
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append('/media/artin/SSD/RESEARCH/PhD/code')
from otherFuncs import smallFuncs
from modelFuncs.Metrics import Precision_Recall_Curve
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


# dir = '/media/artin/SSD/RESEARCH/PhD/vimp2_988_08302013_CB/Prediction/'
# dirM = '/media/artin/SSD/RESEARCH/PhD/vimp2_988_08302013_CB/Label/'

dir = '/array/ssd/msmajdi/experiments/keras/exp6/results/sE12_Cascade_FM00_Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CVs_all/sd2/vimp2_988_08302013_CB/'
dirM = '/array/ssd/msmajdi/experiments/keras/exp6/crossVal/Main/c/vimp2_988_08302013_CB/Label/'
a = smallFuncs.Nuclei_Class(index=1,method='Cascade').All_Nuclei()

ind = 12
msk = nib.load(dir  + a.Names[ind] + '.nii.gz').get_data()
mskM = nib.load(dirM  + a.Names[ind] + '_PProcessed.nii.gz').get_data()

np.unique(msk)
plt.plot(np.unique(msk))
plt.show()

Precision_Recall_Curve(mskM,msk)

ym1 = np.reshape(mskM,[-1,1])
yt1 = np.reshape(msk,[-1,1])

precision, recall, thresholds = precision_recall_curve(yt1, ym1)
average_precision = average_precision_score(yt1, ym1)

print(precision,recall)
print('---')
