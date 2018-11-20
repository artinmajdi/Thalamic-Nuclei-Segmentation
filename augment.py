import numpy as np

def Augment(params , Input):
    print('temporary')

    names = 'normal'
    for AugIx in range(params.augmentLength):
        names = np.append( names , str(AugIx))

    params.augmentDescription = names
    
    return params , Input


def mainAugment(params , Input):

    if params.augment:
        params , Input = Augment(params , Input)
    else:
        Input.Image        = Input.Image[...,np.newaxis]
        Input.CropMask     = Input.CropMask[...,np.newaxis]
        Input.ThalamusMask = Input.ThalamusMask[...,np.newaxis]

        params.augmentDescription = ['normal']

    return params , Input
