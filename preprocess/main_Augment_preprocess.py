import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from augment import augmentMain
from otherFuncs import params
from otherFuncs.smallFuncs import funcExpDirectories

augmentMain( params , 'Linear' )


augmentMain( params , 'NonLinear')
params.directories = funcExpDirectories(params.directories.WhichExperiment)