import os, sys
__file__ = '/array/ssd/msmajdi/code/Thalamus_Keras/mainTest.py'  #! only if I'm using Hydrogen Atom
sys.path.append(os.path.dirname(__file__))
import numpy as np
from otherFuncs import params
import math
import nibabel as nib
from otherFuncs.smallFuncs import correctNumLayers, imageSizesAfterPadding, imShow
import numpy as np

os.path.exists('/array/ssd/msmajdi/code/Thalamus_Keras/mainTest.py')

Subjects = params.directories.Train.Input.Subjects
HardParams = params.directories.Experiment.HardParams

#! checking the number of layers compare to the input image size & fix it if needed
HardParams = correctNumLayers(Subjects, HardParams)

#! Finding the final image sizes after padding & amount of padding
Subjects, HardParams = imageSizesAfterPadding(Subjects, HardParams)


subject = Subjects[list(Subjects)[0]]
im = nib.load(subject.Address + '/' + subject.ImageProcessed + '.nii.gz')
label = nib.load(subject.Label.Address + '/' + subject.Label.LabelProcessed + '.nii.gz').get_data()

imP = np.pad(im, subject.Padding, 'constant')


imShow( im[...,20] , imP[...,20] , im[...,20] )


a = np.zeros((24,10,10))
b = np.zeros((13,10,10))
np.concatenate((a,b),axis=0).shape
