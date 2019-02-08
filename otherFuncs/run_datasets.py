import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs, datasets
from Parameters import paramFunc
from Parameters import UserInfo
from preprocess import applyPreprocess

mode = 'experiment'

params = paramFunc.Run(UserInfo.__dict__)
# params.WhichExperiment.Dataset.CreatingTheExperiment = True


#! reading the user input parameters via terminal
params = smallFuncs.terminalEntries(params)


#! copying the dataset into the experiment folder
datasets.movingFromDatasetToExperiments(params)

# params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)
# params = smallFuncs.inputNamesCheck(params, mode)
# #! needs to run preprocess first to add the PPRocessed.nii.gz files

# #! preprocessing the data
# if params.preprocess.Mode: applyPreprocess.main(params, mode)
# params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)


# #! correcting the number of layers
# num_Layers  = datasets.correctNumLayers(params)
# params.WhichExperiment.HardParams.Model.num_Layers = num_Layers

# #! Finding the final image sizes after padding & amount of padding
# Subjects_Train, Subjects_Test, new_inputSize = datasets.imageSizesAfterPadding(params, mode)


# #! loading the dataset
# Data = datasets.loadDataset(params)

print('---')