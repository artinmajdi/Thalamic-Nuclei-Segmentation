import os
import sys
# sys.path.append(os.path.dirname(__file__))  # 
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
from otherFuncs.datasets import preAnalysis
from otherFuncs import datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess
import tensorflow as tf
from keras import backend as K
import pandas as pd
import xlsxwriter
import csv
import numpy as np
# import json
import nibabel as nib
# from shutil import copyfile , copytree
from tqdm import tqdm
from preprocess import BashCallingFunctionsA, croppingA


dir = '/array/ssd/msmajdi/experiments/keras/exp6/crossVal/Main/a/vimp2_ANON695_03132013/Label/'

a = smallFuncs.Nuclei_Class(index=1,method='Cascade').All_Nuclei()
Volumes = []
for ind , name in zip(a.Indexes , a.Names):
    ms = nib.load(dir + name + '_PProcessed.nii.gz').get_data().sum()
    Volumes.append([ind, ms, ms*(0.5*0.7*0.7)])


Volumes