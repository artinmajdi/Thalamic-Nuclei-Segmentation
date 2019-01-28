import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from augmentA import main_augment
from otherFuncs import smallFuncs


params = smallFuncs.readingTheParams('')[0]

# params = smallFuncs.inputNamesCheck(params, 'experiment')

#! mode: 1: on train & test folders in the experiment
#! mode: 2: on individual image
main_augment( params , 'Linear', 'experiment')
params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)
main_augment( params , 'NonLinear' , 'experiment')
