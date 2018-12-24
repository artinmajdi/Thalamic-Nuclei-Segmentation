import os, sys
__file__ = '/array/ssd/msmajdi/code/Thalamus_Keras/mainTest.py'  #! only if I'm using Hydrogen Atom
sys.path.append(os.path.dirname(__file__))
import numpy as np
from otherFuncs import params
import math
import nibabel as nib
from otherFuncs.smallFuncs import correctNumLayers, imageSizesAfterPadding
import numpy as np


Subjects = params.directories.Train.Input.Subjects
HardParams = params.directories.Experiment.HardParams

#! checking the number of layers compare to the input image size & fix it if needed
HardParams = correctNumLayers(Subjects, HardParams)

#! Finding the final image sizes after padding & amount of padding
Subjects, HardParams = imageSizesAfterPadding(Subjects, HardParams)


subject = Subjects[list(Subjects)[0]]
im = nib.load(subject.Address + '/' + subject.ImageProcessed + '.nii.gz').get_data()
label = nib.load(subject.Label.Address + '/' + subject.Label.LabelProcessed + '.nii.gz').get_data()

im = np.pad(im,subject.Padding, 'constant')


# readingImages()
