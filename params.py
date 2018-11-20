import os
from smallCodes import mkDir , NucleiSelection , checkTestDirectory
from paramsFunctions import whichCropMode , fixDirectoryLastDashSign , funcExpDirectories , augmentLengthChecker

#  ---------------------- Preprocessing Params ----------------------

NucleusName = NucleiSelection(1)

SlicingDirection = 'axial'
SlicingDirection = SlicingDirection.lower()


saveMode = 'nifti'
saveMode = saveMode.lower()

augment = False
augmentLength = 5 
augmentLength = augmentLengthChecker(augment,augmentLength)

normalize = True

normalizeMethod = 'MinMax'
normalizeMethod = normalizeMethod.lower()


Experiment_Number = 1
subExperiment_Number = 1
directories = funcExpDirectories( Experiment_Number , subExperiment_Number)

# default path ; user can change them via terminal by giving other paths
dirrOutput = directories.Results  # should specify by user
dirrInput = directories.Train     # should specify by user


directories.Output = dirrOutput
directories.Input , mt = checkTestDirectory( dirrInput )

MultipleTest = mt
del dirrOutput, dirrInput

# --cropping mode
# 1 or mask:     cropping using the cropped mask acquired from rigid transformation
# 2 or thalamus: cropping using the cropped mask for plain size and Thalamus Prediction for slice numbers
# 3 or both:     cropping using the Thalamus prediction
cropping = True
CroppingMode = 2
CroppingMode = whichCropMode(NucleusName, CroppingMode)  # it changes the mode to 1 if we're analyzing the Thalamus

#  ---------------------- model Params ----------------------
ArchitectureType = 'U-Net'
NumberOfLayers = 3
Optimizer = 'Adam'
