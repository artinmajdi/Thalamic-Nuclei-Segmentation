import os
import sys
import pathlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from otherFuncs import smallFuncs
from modelFuncs.Metrics import Precision_Recall_Curve



dir = 'path-to-case-predictions/'
dirM = 'path-to-case-manual-labels/'

Names = smallFuncs.Nuclei_Class(index=1,method='Cascade').allNames

write_flag = False
PR = {}
if write_flag: df = pd.DataFrame()
if write_flag: writer = pd.ExcelWriter(path=dir + 'Precision_Recall.xlsx', engine='xlsxwriter') 
    
    
for ind in range(13):

    nucleus_name = Names[ind].split('-')[1]
    msk = nib.load(dir  + Names[ind] + '.nii.gz').get_fdata()
    mskM = nib.load(dirM  + Names[ind] + '_PProcessed.nii.gz').get_fdata()

    # plt.plot(np.unique(msk))
    
    precision, recall = Precision_Recall_Curve(y_true=mskM,y_pred=msk, Show=True, name=nucleus_name, directory=dir)

    if write_flag:
        df = pd.DataFrame.from_dict({'precision':precision , 'recall':recall})
        df.to_excel(writer, sheet_name=nucleus_name)

if write_flag: writer.save()

