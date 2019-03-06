import os, sys
sys.path.append(os.path.dirname(os.path.dirname('/array/ssd/msmajdi/code/thalamus/keras/otherFuncs/run_datasets.py')))
from otherFuncs import smallFuncs, datasets
from Parameters import paramFunc
from Parameters import UserInfo
from preprocess import applyPreprocess
import h5py
import numpy as np

mode = 'experiment'

#! reading the user input parameters via terminal
UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)

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
f = h5py.File(params.WhichExperiment.Experiment.address + '/7T_wAug.h5py' , 'w')
for subject in Data.Test:
    f.create_dataset('Test/%s/Image'%(subject),data=Data.Test[subject].Image)
    f.create_dataset('Test/%s/Mask'%(subject),data=Data.Test[subject].Mask)

for subject in Data.Train_ForTest:
    f.create_dataset('Train/%s/Image'%(subject),data=Data.Train_ForTest[subject].Image)
    f.create_dataset('Train/%s/Mask'%(subject),data=Data.Train_ForTest[subject].Mask)

f.visit(print)
f.close()


g = h5py.File(params.WhichExperiment.Experiment.address + '/7T_wAug.h5py' , 'r')
b = g['Train']['vimp2_A']['Image']

b

g.close()














print('----')
