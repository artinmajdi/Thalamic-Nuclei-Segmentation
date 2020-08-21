import os, sys
sys.path.append(os.path.dirname(os.path.dirname('/array/ssd/msmajdi/code/thalamus/keras/otherFuncs/run_datasets.py')))
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo
import preprocess.applyPreprocess as applyPreprocess
import h5py
import numpy as np
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
from tqdm import tqdm
mode = 'experiment'

#! reading the user input parameters via terminal
UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
params = paramFunc.Run(UserInfoB, terminal=False)

#! copying the dataset into the experiment folder
datasets.movingFromDatasetToExperiments(params)

# #! loading the dataset
Data, params = datasets.loadDataset(params)

obj = skimage.measure.regionprops( (Data.Train.Image > 0.5).astype(np.int32) )
ShapeSz = [obj[0].bbox[d+3]-obj[0].bbox[d] for d in range(3)]

a = 2**(params.WhichExperiment.HardParams.Model.num_Layers - 1)
new_inputSize = [  a * np.ceil(s / a) if s % a != 0 else s for s in ShapeSz ]


print('---')
