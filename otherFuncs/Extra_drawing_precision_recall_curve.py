import os
import sys
# sys.path.append(os.path.dirname(__file__))  # 
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append('/media/artin/SSD/RESEARCH/PhD/code')
import otherFuncs.smallFuncs as smallFuncs
from modelFuncs.Metrics import Precision_Recall_Curve
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

dir = '/media/artin/SSD/RESEARCH/PhD/vimp2_988_08302013_CB/Prediction/'
dirM = '/media/artin/SSD/RESEARCH/PhD/vimp2_988_08302013_CB/Label/'
a = smallFuncs.Nuclei_Class(index=1,method='Cascade').All_Nuclei()

msk = nib.load(dir  + '8-Pul.nii.gz').get_data()
mskM = nib.load(dirM  + '8-Pul_PProcessed.nii.gz').get_data()


# plt.plot(np.unique(msk))
plt.plot(np.random.random(10))
plt.show()
Precision_Recall_Curve(mskM,msk)

print('---')
