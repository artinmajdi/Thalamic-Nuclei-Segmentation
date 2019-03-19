import numpy as np
import nibabel as nib
import os, sys
import csv
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
params = paramFunc.Run(UserInfo.__dict__)
import pandas as pd
import h5py
import pickle


Dir = '/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/models/sE555_Hierarchical_CascadewRot7_sd2_Dt0.3/1-THALAMUS'




with open(Dir + '/UserInfoB.pkl', 'rb') as f:
    a = pickle.load(f)
class InputDimensions:
    WoAug = [104, 108, 72]
    wAug = [116, 124, 80]
    wAug_SRI = [116,144,84]
a['InputDimensions']
