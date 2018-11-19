import numpy
import os
import sys
import keras
import nibabel as nib
import numpy as np
import params
import cropping
import augment
import readinginput
from smallCodes import mkDir , listSubFolders
from normalizeInput import normalizeInput


def NucleiSelection(ind):

    if ind == 1:
        NucleusName = '1-THALAMUS'
    elif ind == 2:
        NucleusName = '2-AV'
    elif ind == 4567:
        NucleusName = '4567-VL'
    elif ind == 4:
        NucleusName = '4-VA'
    elif ind == 5:
        NucleusName = '5-VLa'
    elif ind == 6:
        NucleusName = '6-VLP'
    elif ind == 7:
        NucleusName = '7-VPL'
    elif ind == 8:
        NucleusName = '8-Pul'
    elif ind == 9:
        NucleusName = '9-LGN'
    elif ind == 10:
        NucleusName = '10-MGN'
    elif ind == 11:
        NucleusName = '11-CM'
    elif ind == 12:
        NucleusName = '12-MD-Pf'
    elif ind == 13:
        NucleusName = '13-Hb'
    elif ind == 14:
        NucleusName = '14-MTT'

    return NucleusName

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] =  '4'  # 'nan'  #
    UserEntries['IxNuclei'] = [1]
    UserEntries['dataset'] = '20priors'
    UserEntries['method'] = 'old'
    UserEntries['mode'] = 'server'
    UserEntries['onetrain_testIndexes'] = [1]

    for input in sys.argv:

        if input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]

        elif input.split('=')[0] == 'testmode':
            UserEntries['testmode'] = input.split('=')[1] # 'combo' 'onetrain'

        elif input.split('=')[0] == 'dataset':
            UserEntries['dataset'] = input.split('=')[1]

        elif input.split('=')[0] == 'method':
            UserEntries['method'] = input.split('=')[1]

        elif input.split('=')[0] == 'mode':
            UserEntries['mode'] = input.split('=')[1]

        elif input.split('=')[0] == 'nuclei':
            if 'all' in input.split('=')[1]:
                a = range(4,14)
                UserEntries['IxNuclei'] = np.append([1,2,4567],a)

            elif input.split('=')[1][0] == '[':
                B = input.split('=')[1].split('[')[1].split(']')[0].split(",")
                UserEntries['IxNuclei'] = [int(k) for k in B]

            else:
                UserEntries['IxNuclei'] = [int(input.split('=')[1])]

        elif 'onetrain_testIndexes' in input:
            UserEntries['testmode'] = 'onetrain'
            if input.split('=')[1][0] == '[':
                B = input.split('=')[1].split('[')[1].split(']')[0].split(",")
                UserEntries['onetrain_testIndexes'] = [int(k) for k in B]
            else:
                UserEntries['onetrain_testIndexes'] = [int(input.split('=')[1])]

    return UserEntries

def funcFlipLR_UD(im):
    for i in range(im.shape[2]):
        im[...,i] = np.fliplr(im[...,i])
        im[...,i] = np.flipud(im[...,i])

    return im

FullData = readinginput.mainloadingImage(params)

for ind in [1]: # range(len(FullData)):

    im           = FullData[ind].Image.get_data()
    CropMask     = FullData[ind].CropMask.get_data()
    ThalamusMask = FullData[ind].ThalamusMask
    TestAddress  = FullData[ind].TestAddress

    if params.OptionNormalize:
        im = normalizeInput(im , params)

    if params.OptionAugment:
        imA , CropMaskA , ThalamusMaskA = augment.main(im , CropMask , ThalamusMask)

    for aIx in range(imA.shape[3]):

        im           = imA[...,aIx]
        CropMask     = CropMaskA[...,aIx]
        ThalamusMask = ThalamusMaskA.copy()

        imCropped , CropCordinates = cropping.mainCropping(params.CroppingMode , im , CropMask , ThalamusMask)


print('finished')
params.Directory_Experiment
params.Directory_input
