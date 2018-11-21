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


Dir = '/array/ssd/msmajdi/experiments/Keras/vimp2_test'
import os
os.stat(Dir + '/WMnMPRAGE.nii.gz')

### reading from text file
params = terminalEntries(params)

BiasCorrection(params.directories.Input.Address , params.directories.Input.Files.origImage)
# RigidRegistration(params)

#### check image format and convert to nifti
params
### fix the reading Image function
Input  = mainloadingImage(params.directories.Input.Address , params.directories.Input.Files)

print('---')
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
params.directories.Input.Files.BiasCorrected
params.directories.Input.Files.Crop
params.directories.Output.Result.