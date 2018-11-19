import numpy as np

def main(im , CropMask , ThalamusMask):
    return im[...,np.newaxis] , CropMask[...,np.newaxis] , ThalamusMask
