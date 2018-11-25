import os
import sys
# import keras
import nibabel as nib
import numpy as np
import params
import cropping
from augment import augmentMain
from BashCallingFunctions import RigidRegistration, Cropping, BiasCorrection
from readinginput import mainloadingImage
from smallCodes import mkDir , listSubFolders , choosingSubject , NucleiSelection , terminalEntries , checkInputDirectory, funcExpDirectories# , checkingSubFolders
from normalizeInput import normalizeMain

###  check 3T 7T dimension and interpolation
#### check image format and convert to nifti

subj = params.directories.Train.Input.Subjects

params = terminalEntries(params)

for mode in ['Train','Test']:

    dirr = params.directories.Train if mode == 'Train' else params.directories.Test
    for sj in dirr.Input.Subjects :
        subject = dirr.Input.Subjects[sj]
        print(mode, 'BiasCorrection: ',sj)
        BiasCorrection( subject)

augmentMain( params , 'Linear')
for mode in ['Train','Test']:

    params.directories = funcExpDirectories(params.directories.Experiment)
    dirr = params.directories.Train if mode == 'Train' else params.directories.Test
    for sj in dirr.Input.Subjects :
        subject = dirr.Input.Subjects[sj]

        print(mode, 'RigidRegistration: ',sj)
        RigidRegistration( subject , params.directories.Experiment.HardParams.Template)

        print(mode, 'Cropping: ',sj)
        Cropping( subject )

augmentMain( params , 'NonLinear')



    # params.directories = funcExpDirectories(params.directories.Experiment)
    # for ind in range(len(subDirectories)):

    #     params.directories.Input = checkInputDirectory( subDirectories[ind] , params.TrainParams.NucleusName )
    #     Input  = mainloadingImage(params.directories.Input)




print('finished')
