import numpy as np
import pandas as pd
import os
import smallFuncs
import nibabel as nib
from collections import defaultdict
direcotry_main = '/array/ssd/msmajdi/experiments/keras/exp6/results/'
nuclei = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Names

Dices = {}
metric = 'HD'
for x in 'a b c'.split():
    direcotry = direcotry_main + f'sE12_Cascade_FM00_Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_7T_Init_Rn_test_ET_3T_CV_{x}/2.5D_MV/'

    subjects = [s for s in os.listdir(direcotry) if 'vimp' in s]
    for subj in subjects:
        df  = pd.read_csv(direcotry + subj + '/' + metric + '_All.txt',delimiter=' ',header=None, index_col=0, names=[subj]) # 
        Dices[subj] = list(df[subj])

        print(subj)

a = pd.DataFrame.from_records(Dices,index=nuclei)
a.T.to_csv(metric + '_ET_7T_3T.csv')


# a = nib.viewers.OrthoSlicer3D(fitlered_prediction)

# b = nib.viewers.OrthoSlicer3D(im)
# b.link_to(a)
# b.show()
