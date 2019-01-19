import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs, datasets
from Parameters import paramFunc
from Parameters import UserInfo
from preprocess import applyPreprocess

mode = 'experiment'

params = paramFunc.Run(UserInfo.__dict__)
params.preprocess.Mode = False
params.preprocess.CreatingTheExperiment = True


#! reading the user input parameters via terminal
params = smallFuncs.terminalEntries(params)


#! copying the dataset into the experiment folder
if not (os.path.exists(params.directories.Train.address) and os.path.exists(params.directories.Test.address)): 
    datasets.movingFromDatasetToExperiments(params)

params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)
params = smallFuncs.inputNamesCheck(params, mode)
#! needs to run preprocess first to add the PPRocessed.nii.gz files

#! preprocessing the data
if params.preprocess.Mode: applyPreprocess.main(params, mode)
params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)


#! correcting the number of layers
num_Layers, params = smallFuncs.correctNumLayers(params)


#! Finding the final image sizes after padding & amount of padding
Subjects_Train, Subjects_Test, new_inputSize, params = smallFuncs.imageSizesAfterPadding(params, mode)


#! loading the dataset
Data = datasets.loadDataset(params)

print('---')