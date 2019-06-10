import nibabel as nib
import numpy as np
# import nilearn
# import matplotlib.pyplot as plt
# import os, sys
from skimage import measure
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# import otherFuncs.smallFuncs as smallFuncs
# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# params = paramFunc.Run(UserInfo.__dict__, terminal=False)
from scipy import misc
import skimage


dir = '/home/artinl/Documents/vimp2_0944_07092014_SRI/'


im = nib.load(dir + 'PProcessed.nii.gz')
Image = im.get_data()

Mask = nib.load(dir + 'Label/1-THALAMUS_PProcessed.nii.gz').get_data()

sz = Image.shape
newShape =  (2*sz[0] , 2*sz[1]) + (sz[2],)  


# Image2 = np.zeros(newShape)
# Mask2 = np.zeros(newShape)
# for i in range(Image.shape[2]):
#     Image2[...,i] = misc.imresize(Image[...,i] ,size=newShape[:2] , interp='cubic')
#     Mask2[...,i]  = misc.imresize( (Mask[...,i] > 0.5).astype(np.float32) ,size=newShape[:2] , interp='bilinear')

sz = Image.shape
newShape =  (2*sz[0] , 2*sz[1]) + (sz[2],)  

Image3 = np.zeros(newShape)
Mask3  = np.zeros(newShape)
tform = skimage.transform.AffineTransform(scale=(2,2) )
for i in range(Image.shape[2]):
    Image3[...,i] = skimage.transform.warp( Image[...,i], tform.inverse, output_shape=newShape[:2], order=3)
    Mask3[...,i]  = skimage.transform.warp( (Mask[...,i] > 0.5).astype(np.float32) ,  tform.inverse, output_shape=newShape[:2], order=0)


# a = nib.viewers.OrthoSlicer3D(Image)
a = nib.viewers.OrthoSlicer3D(Image , title='orig')
b = nib.viewers.OrthoSlicer3D(Image2, title='Mask2')
c = nib.viewers.OrthoSlicer3D(Image3, title='Mask3')

a.link_to(b)
a.link_to(c)
a.show()
# b.show()


