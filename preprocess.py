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
from smallCodes import mkDir , listSubFolders , choosingSubject , NucleiSelection , terminalEntries , checkInputDirectory, checkingSubFolders
from normalizeInput import normalizeMain

###  check 3T 7T dimension and interpolation
#### check image format and convert to nifti



params = terminalEntries(params)


params = checkingSubFolders(params)
for add in params.directories.Input.Address:


    params.directories.Input = checkInputDirectory( add , params.TrainParams.Nucleus.Name )
    BiasCorrection( params.directories.Input )
    augmentMain(params , 'Linear')


params = checkingSubFolders(params)
for add in params.directories.Input.Address:

    params.directories.Input = checkInputDirectory( add , params.TrainParams.Nucleus.Name )
    RigidRegistration( params.directories.Input , params.directories.Template )
    Cropping( params.directories.Input)
    augmentMain(params , 'NonLinear')


subDirectories = os.listdir(params.directories.Input.Address)
# for ind in range(len(subDirectories)):

#     params.directories.Input = checkInputDirectory( subDirectories[ind] , params.TrainParams.NucleusName )
#     Input  = mainloadingImage(params.directories.Input)




print('finished')
