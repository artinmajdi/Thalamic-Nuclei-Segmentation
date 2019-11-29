import os
import sys
# sys.path.append(os.path.dirname(__file__))  # 
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
from modelFuncs.Metrics import Precision_Recall_Curve
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

dir = '/array/ssd/msmajdi/experiments/keras/exp7/crossVal/temp/Main/a/vimp2_967_08132013_KW/left/sd2/'
dirM = '/array/ssd/msmajdi/experiments/keras/exp7/crossVal/temp/Main/a/vimp2_967_08132013_KW/Label/'
a = smallFuncs.Nuclei_Class(index=1,method='Cascade').All_Nuclei()

ind = 12
msk = nib.load(dir  + a.Names[ind] + '.nii.gz').get_data()
mskM = nib.load(dirM  + a.Names[ind] + '_PProcessed.nii.gz').get_data()


plt.plot(np.unique(msk))
# plt.show()
Precision_Recall_Curve(mskM,msk)

ym1 = np.reshape(mskM,[-1,1])
yt1 = np.reshape(msk,[-1,1])

precision, recall, thresholds = precision_recall_curve(yt1, ym1)
average_precision = average_precision_score(yt1, ym1)

print(precision,recall)
print('---')
