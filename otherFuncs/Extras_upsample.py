# import nilearn
import nibabel as nib
import keras
import tensorflow as tf
import os,sys
from skimage.transform import AffineTransform , warp
import numpy as np
# import smallFuncs
from nilearn import image as NLimage
from scipy import ndimage



def saveImage(image , affine , header , outDirectory):
    def mkDir(Dir):
        if not os.path.isdir(Dir): os.makedirs(Dir)
        return Dir
            
    mkDir(os.path.dirname(outDirectory))
    out = nib.Nifti1Image((image).astype('float32'),affine)
    out.get_header = header
    nib.save(out , outDirectory)

def upsample_Image(Mask=[] , scale=2):
    szM = Mask.shape
    
    Mask_US  = np.zeros( (szM[0] , scale*szM[1] , scale*szM[2])  )
    newShape = (scale*szM[1] , scale*szM[2])
    tform = AffineTransform(scale=(scale, scale))
    for ch in range(Mask.shape[0]):
        Mask_US[ch, : ,:] = warp( np.squeeze(Mask[ch, : ,:]).astype(np.float32) ,  tform.inverse, output_shape=newShape, order=0)
    
    return Mask

def closeMask(mask):
    struc = ndimage.generate_binary_structure(3,2)
    maskD = ndimage.binary_closing(mask.get_data(), structure=struc)
    return nib.Nifti1Image(maskD, affine=Mask.affine , header=Mask.header)

def upsample_Image_nilearn(Mask=[], scale=1):
    affine = Mask.affine.copy()
    for i in range(3): affine[i,i] /= scale
    return NLimage.resample_img(img=Mask , target_affine=affine , interpolation='nearest')

dir2 = '/home/artinl/Documents/RESEARCH/Results/for grant/988_wmn/Label/'
Mask = NLimage.load_img(dir2 + 'AllLabels_new.nii.gz')

upsample_Image_nilearn(Mask=Mask, scale=2)






# a = nib.viewers.OrthoSlicer3D(Mask.get_data())
# b = nib.viewers.OrthoSlicer3D(Mask_US) # .get_data()
# a.show()

# nib.save(Mask_US, dir2 + 'AllLabels_US.nii.gz')
