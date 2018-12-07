import os
import sys
# import keras
import nibabel as nib
import numpy as np
import params
import cropping
from augment import augmentMain
from BashCallingFunctions import RigidRegistration, Bash_Cropping, BiasCorrection
from readinginput import mainloadingImage
from smallFuncs import mkDir , listSubFolders , choosingSubject , NucleiSelection , terminalEntries , checkInputDirectory, funcExpDirectories , inputNamesCheck
from normalizeInput import normalizeMain

###  check 3T 7T dimension and interpolation
#### check image format and convert to nifti

params = terminalEntries(params)
params = inputNamesCheck(params)


for mode in ['Train','Test']:
 
    if params.preprocess.TestOnly and 'Train' in mode:
        continue

    dirr = params.directories.Train if mode == 'Train' else params.directories.Test     
    for sj in dirr.Input.Subjects :
        subject = dirr.Input.Subjects[sj]
        print(mode, 'BiasCorrection: ',sj)
        BiasCorrection( subject , params)

params.directories = funcExpDirectories(params.directories.Experiment)
augmentMain( params , 'Linear' )
params.directories = funcExpDirectories(params.directories.Experiment)
for mode in ['Train','Test']:

    dirr = params.directories.Train if mode == 'Train' else params.directories.Test
    for sj in dirr.Input.Subjects :
        subject = dirr.Input.Subjects[sj]

        print(mode, 'RigidRegistration: ',sj)
        RigidRegistration( subject , params.directories.Experiment.HardParams.Template , params.preprocess)

        print(mode, 'Cropping: ',sj)
        Bash_Cropping( subject , params)

augmentMain( params , 'NonLinear')
params.directories = funcExpDirectories(params.directories.Experiment)



    # params.directories = funcExpDirectories(params.directories.Experiment)
    # for ind in range(len(subDirectories)):

    #     params.directories.Input = checkInputDirectory( subDirectories[ind] , params.TrainParams.NucleusName )
    #     Input  = mainloadingImage(params.directories.Input)




print('finished')
