# import nilearn
import nibabel as nib
import keras
import tensorflow as tf
from skimage.transform import AffineTransform , warp
import numpy as np
from nilearn import image as imageNilearn


dir = '/home/artinl/Documents/RESEARCH/Results/for grant/988_wmn/'

def saveImage(Image , Affine , Header , outDirectory):
    mkDir(outDirectory.split(os.path.basename(outDirectory))[0])
    out = nib.Nifti1Image((Image).astype('float32'),Affine)
    out.get_header = Header
    nib.save(out , outDirectory)

def upsample_Image(Mask , scale):
    szM = Mask.shape
    
    Mask  = np.zeros( (szM[0] , scale*szM[1] , scale*szM[2] , szM[3])  )

    newShape = (scale*szM[1] , scale*szM[2])

    # for i in range(Image.shape[2]):
    #     Image2[...,i] = scipy.misc.imresize(Image[...,i] ,size=newShape[:2] , interp='cubic')
    #     Mask2[...,i]  = scipy.misc.imresize( (Mask[...,i] > 0.5).astype(np.float32) ,size=newShape[:2] , interp='bilinear')

    tform = AffineTransform(scale=(scale, scale))
    for ch in range(Mask.shape[3]):
        Mask[: ,: ,ch]  = warp( (np.squeeze(Mask[: ,: ,ch]) > 0.5).astype(np.float32) ,  tform.inverse, output_shape=newShape, order=0)
    
    return Mask


dir2 = dir + 'predictions/left/2.5D_MV/'
Mask = imageNilearn.load_img(dir2 + 'AllLabels.nii.gz')

affine = Mask.affine
for i in range(3): affine[i,i] *= 2

affine

Mask_US = imageNilearn.resample_img(img=Mask , target_affine=affine,interpolation='nearest')
# Mask.shape

a = nib.viewers.OrthoSlicer3D(Mask.get_data())
b = nib.viewers.OrthoSlicer3D(Mask_US.get_data())

a.show()

# nib.save(Mask_US, dir2 + 'AllLabels_US.nii.gz')

print('----')