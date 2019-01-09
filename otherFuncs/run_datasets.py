import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs, datasets
from Parameters import params
params.preprocess.Mode = False
params.preprocess.CreatingTheExperiment = False
from preprocess import applyPreprocess

mode = 'experiment'


#! reading the user input parameters via terminal
params = smallFuncs.terminalEntries(params)


#! copying the dataset into the experiment folder
if params.preprocess.CreatingTheExperiment: datasets.movingFromDatasetToExperiments(params)


#! preprocessing the data
if params.preprocess.Mode: applyPreprocess.main(params, mode)
params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)


#! correcting the number of layers
params = smallFuncs.correctNumLayers(params)


#! Finding the final image sizes after padding & amount of padding
params = smallFuncs.imageSizesAfterPadding(params, mode)


#! loading the dataset
Data, params = datasets.loadDataset(params)

print('---')