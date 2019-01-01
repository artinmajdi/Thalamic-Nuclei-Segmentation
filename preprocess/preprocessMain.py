import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import nibabel as nib
import numpy as np
from otherFuncs import params
params.preprocess.Mode = True
from preprocess import cropping
from preprocess.augment import augmentMain
from preprocess.BashCallingFunctions import RigidRegistration, Bash_Cropping, BiasCorrection
from otherFuncs.smallFuncs import terminalEntries, funcExpDirectories, inputNamesCheck
from preprocess.normalizeMine import normalizeMain

###  check 3T 7T dimension and interpolation
#### check image format and convert to nifti


params = terminalEntries(params)
params = inputNamesCheck(params)


for mode in ['Train','Test']:

    if params.preprocess.TestOnly and 'Train' in mode:
        continue

    dirr = params.directories.Train if mode == 'Train' else params.directories.Test
    for sj in dirr.Input.Subjects:
        subject = dirr.Input.Subjects[sj]
        print(mode, 'BiasCorrection: ',sj)
        BiasCorrection( subject , params)

params.directories = funcExpDirectories(params.directories.WhichExperiment)
augmentMain( params , 'Linear' )
params.directories = funcExpDirectories(params.directories.WhichExperiment)
for mode in ['Train','Test']:

    dirr = params.directories.Train if mode == 'Train' else params.directories.Test
    for sj in dirr.Input.Subjects :
        subject = dirr.Input.Subjects[sj]

        print(mode, 'RigidRegistration: ',sj)
        RigidRegistration( subject , params.directories.WhichExperiment.HardParams.Template , params.preprocess)

        print(mode, 'Cropping: ',sj)
        Bash_Cropping( subject , params)

augmentMain( params , 'NonLinear')
