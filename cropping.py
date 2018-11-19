import numpy as np


def mainCropping(mode , im , CropMask , ThalamusMask):

    if mode == 1:
        gap=[0,0,1]
        im , CropCordinates = funcCropping_Mode1(im , CropMask,gap)

    elif mode == 2:
        gap=[20,20,1]
        im , CropCordinates = funcCropping_Mode1(im , ThalamusMask,gap)

    elif mode == 3:
        gap=[0,0,1]
        im , CropCordinates = funcCropping_Mode3_SlicingFromThalamus(im , CropMask , ThalamusMask , gap)

    return im , CropCordinates


def cropFunc(im , c1,c2,c3 , gap):

    d1 = [  c1[0]-gap[0] , c1[ c1.shape[0]-1 ]+gap[0]  ]
    d2 = [  c2[0]-gap[1] , c2[ c2.shape[0]-1 ]+gap[1]  ]
    SN = [  c3[0]-gap[2] , c3[ c3.shape[0]-1 ]+gap[2]  ]
    SliceNumbers = range(SN[0],SN[1])

    im = im[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers ]
    CropCordinates = [d1,d2,SliceNumbers]

    return im , CropCordinates

def funcCropping_Mode1(im , CropMask , gap):

    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 10)[0]
    c2 = np.where(np.sum(ss,axis=0) > 10)[0]

    ss = np.sum(CropMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 10)[0]

    im , CropCordinates = cropFunc(im , c1,c2,c3 , gap)

    return im , CropCordinates

def funcCropping_Mode3_SlicingFromThalamus(im , CropMask , ThalamusMask , gap):

    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 1)[0]
    c2 = np.where(np.sum(ss,axis=0) > 1)[0]

    ss = np.sum(ThalamusMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 1)[0]

    im , CropCordinates = cropFunc(im , c1,c2,c3 , gap)

    return im , CropCordinates
