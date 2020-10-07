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
import pandas as pd


dir = '/array/ssd/msmajdi/experiments/keras/exp6/results/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/sd2/vimp2_967_08132013_KW/'
dirM = '/array/ssd/msmajdi/experiments/keras/exp6/crossVal/Main/a/vimp2_967_08132013_KW/Label/'

Names = smallFuncs.Nuclei_Class(index=1,method='Cascade').allNames

write_flag = False
PR = {}
if write_flag: df = pd.DataFrame()
if write_flag: writer = pd.ExcelWriter(path=dir + 'Precision_Recall.xlsx', engine='xlsxwriter') 
    
    
for ind in range(13):

    nucleus_name = Names[ind].split('-')[1]
    msk = nib.load(dir  + Names[ind] + '.nii.gz').get_data()
    mskM = nib.load(dirM  + Names[ind] + '_PProcessed.nii.gz').get_data()

    # plt.plot(np.unique(msk))
    
    precision, recall = Precision_Recall_Curve(y_true=mskM,y_pred=msk, Show=True, name=nucleus_name, directory=dir)

    if write_flag:
        df = pd.DataFrame.from_dict({'precision':precision , 'recall':recall})
        df.to_excel(writer, sheet_name=nucleus_name)

if write_flag: writer.save()

