import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import os
import pathlib

directory = str(pathlib.Path(__file__).parent) + '/'

im = nib.load(directory + 'origtemplate.nii.gz')
cord = tuple([range(83, 176), range(160, 251), range(95, 151)])

tempMsk1 = np.zeros(im.shape) > 0
tempMsk2 = np.zeros(im.shape) > 0
tempMsk3 = np.zeros(im.shape) > 0

tempMsk1[cord[0], :, :] = True
tempMsk2[:, cord[1], :] = True
tempMsk3[:, :, cord[2]] = True

cropMask = tempMsk1 * tempMsk2 * tempMsk3
sumMask = tempMsk1 + tempMsk2 + tempMsk3

ind = 140

maskF2 = nib.Nifti1Image(cropMask.astype(np.float32), im.affine)
maskF2.get_header = im.header
nib.save(maskF2, directory + 'CropMaskV3.nii.gz')

