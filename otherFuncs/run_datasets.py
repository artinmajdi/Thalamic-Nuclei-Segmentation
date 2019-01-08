import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import params, smallFuncs, datasets
params.preprocess.Mode = False
params.preprocess.CreatingTheExperiment = False
from preprocess.preprocessA import main_preprocess
mode = 'experiment'


#! reading the user input parameters via terminal
params = smallFuncs.terminalEntries(params)


#! copying the dataset into the experiment folder
if params.preprocess.CreatingTheExperiment: datasets.movingFromDatasetToExperiments(params)


#! preprocessing the data
if params.preprocess.Mode: main_preprocess(params, mode)
params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)


#! correcting the number of layers
params = smallFuncs.correctNumLayers(params)


#! Finding the final image sizes after padding & amount of padding
params = smallFuncs.imageSizesAfterPadding(params, mode)


#! loading the dataset
Data,Data2,  params = datasets.loadDataset(params)

print('---')