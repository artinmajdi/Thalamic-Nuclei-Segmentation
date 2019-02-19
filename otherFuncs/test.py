import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
from skimage import morphology
from skimage.measure import regionprops, label
import os

float(0.001)
isinstance([2],list)

def dilateMask(mask, gapDilation):
    struc = ndimage.generate_binary_structure(3,2)
    struc = ndimage.iterate_structure(struc,gapDilation)
    return ndimage.binary_dilation(mask, structure=struc)

def func_CropCoordinates(CropMask):
    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 10)[0]
    c2 = np.where(np.sum(ss,axis=0) > 10)[0]

    ss = np.sum(CropMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 10)[0]

    BBCord = [   [c1[0],c1[-1]]  ,  [c2[0],c2[-1]]  , [c3[0],c3[-1]]  ]

    return BBCord

def saveImage(Image , Affine , Header , outDirectory):
    out = nib.Nifti1Image((Image).astype('float32'),Affine)
    out.get_header = Header
    nib.save(out , outDirectory)

def MyShow(imD,imDilated,ind):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(imD[ind,...],cmap='gray')
    axs[1].imshow(imDilated[ind,...],cmap='gray')

dir = '/array/ssd/msmajdi/experiments/keras/exp6_cascade_7T/results/subExp2_Loss_BCE/TrainData_Output/'
lst = os.listdir(dir)

ind = 2
lst[ind]
imD = nib.load(dir + lst[ind] + '/1-THALAMUS.nii.gz').get_data()
# imD = morphology.closing(label(imD))
objects = regionprops(label(imD))
area = []
for obj in objects: area = np.append(area, obj.area)

Ix = np.argsort(area)
objects[ Ix[-1] ].bbox

func_CropCoordinates(imD)


ss = np.sum(imDilated,axis=2)
plt.plot(np.sum(ss,axis=1))

func_CropCoordinates(imD)
func_CropCoordinates(imDilated)

saveImage(imDilated , im.affine , im.header , dir + '1-THALAMUS_Dilated.nii.gz')
