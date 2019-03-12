import numpy as np
import nibabel as nib
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from Parameters import UserInfo, paramFunc
from otherFuncs import smallFuncs
import pickle


Dir = '/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/models/subExp2_MinMax_Cascade_wAug_Loss_BCE_nl3/1-THALAMUS/hist_history.pkl'
with open(Dir,"rb") as f:
    data = pickle.load(f)

data.keys()
