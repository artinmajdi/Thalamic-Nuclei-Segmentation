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
from smallCodes import mkDir , listSubFolders , choosingSubject , NucleiSelection , terminalEntries , checkInputDirectory
from normalizeInput import normalizeMain

### reading from text file
params = terminalEntries(params)

#### check image format and convert to nifti

BiasCorrection( params.directories.Input.Address , params.directories.Input.Files.origImage )

RigidRegistration( params.directories.Input , params.directories.Template )

params.directories.Input = checkInputDirectory( params.directories.Input.Address )
Input  = mainloadingImage(params.directories.Input.Address , params.directories.Input.Files , params.TrainParams.NucleusName)

# for ind in range(len(Input)):

    ###  check 3T 7T dimension and interpolation
#     RigidRegistration(params)
    # Cropping(params)

    # Input[ind] = normalizeMain(params , Input[ind])

    ### finish augmenting
    # if params.augment.mode and (params.augment.Rotation or params.augment.Shift):
    #     params , Input[ind] = augmentMain(params , Input[ind])

    # Input[ind] , CropCordinates = cropping.mainCropping( params , Input[ind] )

    # if params.augment.mode and params.augment.NonRigidWarp:
    #     params , Input[ind] = augmentMain(params , Input[ind])

print('finished')
# params.Directory_Experiment
# params.Directory_input
