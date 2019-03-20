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
mode = 'experiment'

#! reading the user input parameters via terminal
UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
UserInfoB['simulation'].slicingDim = [2]
UserInfoB['simulation'].nucleus_Index = [1]
params = paramFunc.Run(UserInfoB)
# params.WhichExperiment.Dataset.CreatingTheExperiment = True

#! copying the dataset into the experiment folder
datasets.movingFromDatasetToExperiments(params)

# params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
# params = smallFuncs.inputNamesCheck(params, mode)
# #! needs to run preprocess first to add the PPRocessed.nii.gz files

# #! preprocessing the data
# if params.preprocess.Mode: applyPreprocess.main(params, mode)
# params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)

# #! loading the dataset
Data, params = datasets.loadDataset(params)

