import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from augmentA import main_augment
from otherFuncs import params, smallFuncs
params.preprocess.Mode = True

#! mode: 1: on train & test folders in the experiment
#! mode: 2: on individual image
params = smallFuncs.terminalEntries(params)
main_augment( params , 'Linear', 'experiment')
params.directories = smallFuncs.funcExpDirectories(params.directories.WhichExperiment)
main_augment( params , 'NonLinear' , 'experiment')
