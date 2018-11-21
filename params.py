import os
from smallCodes import NucleiSelection , whichCropMode , fixDirectoryLastDashSign , funcExpDirectories , augmentLengthChecker , InputNames , checkInputDirectory , checkOutputDirectory

#  ---------------------- model Params ----------------------
class TrainParams:
    ArchitectureType = 'U-Net'
    NumberOfLayers = 3
    Optimizer = 'Adam'
    NucleusName = NucleiSelection(1)

class augment:
    mode = False
    augmentLength = 5
    Rotation = False
    Shift = False
    NonRigidWarp = True

augment = augmentLengthChecker(augment)

class normalize:
    mode = True
    method = 'MinMax'

normalize.method = normalize.method.lower()

class inputImageParams:
    SlicingDirection = 'axial'
    saveMode = 'nifti'
    Experiment_Number = 1
    subExperiment_Number = 1

inputImageParams.SlicingDirection = inputImageParams.SlicingDirection.lower()
inputImageParams.saveMode = inputImageParams.saveMode.lower()

directories = funcExpDirectories( inputImageParams.Experiment_Number , inputImageParams.subExperiment_Number)

# default path ; user can change them via terminal by giving other paths
if 1:
    user_output = '/array/ssd/msmajdi/experiments/Keras/vimp2_test' # directories.Results  # should specify by user
    directories.Output = checkOutputDirectory(user_output , inputImageParams.subExperiment_Number)
    del user_output

if 1:
    user_Input = '/array/ssd/msmajdi/experiments/Keras/vimp2_test' # directories.Train     # should specify by user
    directories.Input = checkInputDirectory( user_Input )
    del user_Input

# --cropping mode
# 1 or mask:     cropping using the cropped mask acquired from rigid transformation
# 2 or thalamus: cropping using the cropped mask for plain size and Thalamus Prediction for slice numbers
# 3 or both:     cropping using the Thalamus prediction
method = 2
class cropping:
    mode = True
    method = whichCropMode(TrainParams.NucleusName, method)  # it changes the mode to 1 if we're analyzing the Thalamus

del method


class preprocess:
    augment = augment
    cropping = cropping
    normalize = normalize

del augment, cropping, normalize
