import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np


def chooseAugmented(Input , AugIx):
    return Input.Image[...,AugIx], Input.CropMask[...,AugIx], Input.ThalamusMask[...,AugIx]

def main_cropping(params , Input):

    if params.cropping:
        CropCordinatesAll = np.zeros((Input.Image.shape[3]))
        for AugIx in range(Input.Image.shape[3]):

            Image, CropMask, ThalamusMask = chooseAugmented( Input , AugIx )

            if (params.CroppingMode == 1) | ('mask' in params.CroppingMode ):
                Gap=[0,0,1]
                Image , CropCordinates = funcCropping_Mode( Image , CropMask , Gap )

            elif (params.CroppingMode == 2) | ('thalamus' in params.CroppingMode ):
                Gap=[20,20,1]
                Image , CropCordinates = funcCropping_Mode( Image , ThalamusMask , Gap )

            elif (params.CroppingMode == 3) | ('both' in params.CroppingMode ):
                Gap=[0,0,1]
                Image , CropCordinates = funcCropping_Mode3_SlicingFromThalamus( Image , CropMask , ThalamusMask , Gap )

            Input.Image[...,AugIx] = Image
            CropCordinatesAll[AugIx] = CropCordinates

    return Input , CropCordinatesAll


def cropFunc(im , c1,c2,c3 , Gap):

    d1 = [  c1[0]-Gap[0] , c1[ c1.shape[0]-1 ]+Gap[0]  ]
    d2 = [  c2[0]-Gap[1] , c2[ c2.shape[0]-1 ]+Gap[1]  ]
    SN = [  c3[0]-Gap[2] , c3[ c3.shape[0]-1 ]+Gap[2]  ]
    SliceNumbers = range(SN[0],SN[1])

    im = im[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers ]
    CropCordinates = [d1,d2,SliceNumbers]

    return im , CropCordinates

def funcCropping_Mode(im , CropMask , Gap):

    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 10)[0]
    c2 = np.where(np.sum(ss,axis=0) > 10)[0]

    ss = np.sum(CropMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 10)[0]

    im , CropCordinates = cropFunc(im , c1,c2,c3 , Gap)

    return im , CropCordinates

def funcCropping_Mode3_SlicingFromThalamus(im , CropMask , ThalamusMask , Gap):

    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 1)[0]
    c2 = np.where(np.sum(ss,axis=0) > 1)[0]

    ss = np.sum(ThalamusMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 1)[0]

    im , CropCordinates = cropFunc(im , c1,c2,c3 , Gap)

    return im , CropCordinates
