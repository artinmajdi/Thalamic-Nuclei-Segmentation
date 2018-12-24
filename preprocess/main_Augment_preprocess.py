import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from augment import augmentMain
import params
from otheruncs.smallFuncs import funcfuncExpDirectories

augmentMain( params , 'Linear' )


augmentMain( params , 'NonLinear')
params.directories = funcExpDirectories(params.directories.Experiment)