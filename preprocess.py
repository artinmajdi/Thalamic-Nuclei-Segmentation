import os
import sys
# import keras
import nibabel as nib
import numpy as np
import params
import cropping
import augment
import readinginput
from smallCodes import mkDir , listSubFolders , choosingSubject , NucleiSelection , terminalEntries
from paramsFunctions import checkTestDirectory
from normalizeInput import normalizeMain
from numpy.lib.recfunctions import append_fields


params = terminalEntries(params)
Input  = readinginput.mainloadingImage(params)


for ind in range(len(Input)):

    Input[ind] = normalizeMain(params , Input[ind])

    params , Input[ind] = augment.mainAugment(params , Input[ind])

    Input[ind] , CropCordinates = cropping.mainCropping( params , Input[ind] )


print('finished')
# params.Directory_Experiment
# params.Directory_input
