import numpy as np
import os 


def LinearFunc(params):

    names = 'normal'
    for AugIx in range(params.augmentLength):
        names = np.append( names , str(AugIx))

    params.augmentDescription = names


def NonLinearFunc(params):

    Image = 'crop_LIFUP004_MPRAGE_WMn.nii.gz'
    Reference = '../case3/crop_LIFUP003_MPRAGE_WMn.nii.gz'
    Output = 'aug1.nii.gz'
    os.system("ANTS 3 -m CC[%s, %s,1,5] -t SyN[0.25] -r Gauss[3,0] -o test -i 30x90x20 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000"%(Image , Reference) )
    os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t testWarp.nii.gz"%(Image , Output , Image) )


def augmentMain(params , Flag):

    if params.augment.mode and (params.augment.Rotation or params.augment.Shift) and (Flag == 'Linear'):
        # LinearFunc(params)
        print('---')
    elif params.augment.mode and params.augment.NonRigidWarp and (Flag == 'NonLinear'):
        # NonLinearFunc(params)
        print('---')
        