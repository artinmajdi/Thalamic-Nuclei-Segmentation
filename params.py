import os
from smallCodes import NucleiSelection , whichCropMode , fixDirectoryLastDashSign , funcExpDirectories , augmentLengthChecker , InputNames , checkInputDirectory , checkOutputDirectory

#  ---------------------- model Params ----------------------



class ImageModelParams:
    SlicingDirection = 'axial'.lower()
    SaveMode = 'nifti'.lower()
    ArchitectureType = 'U-Net'
    NumberOfLayers = 3
    Optimizer = 'Adam'

Experiment_Number = 1
SubExperiment_Number = 1
class Experiment:
    Experiment = 'Experiment' + str(Experiment_Number) + ''
    SubExperiment = 'SubExperiment' + str(SubExperiment_Number) + ''
    AllExperiments = '/array/ssd/msmajdi/experiments/Keras'

class Nucleus:
    Index = [1]
    Name = NucleiSelection( Index[0] )

class TrainParams:
    Experiment = Experiment
    WhichMachine = 'server'
    GPU_Index = '1'
    ImageModelParams = ImageModelParams
    Nucleus = Nucleus

del ImageModelParams, Nucleus, Experiment, Experiment_Number, SubExperiment_Number


class Reference:
    Name = ''
    Address = ''

class Augment:
    Mode = False
    AugmentLength = 5
    Rotation = False
    Shift = False
    NonRigidWarp = True
    Reference = Reference

del Reference

Augment = augmentLengthChecker(Augment)

class Normalize:
    Mode = True
    Method = 'MinMax'.lower()

directories = funcExpDirectories( TrainParams.Experiment , TrainParams.Nucleus.Name)

# default path ; user can change them via terminal by giving other paths
if 1:
    user_Input = '/array/ssd/msmajdi/experiments/Keras/vimp2_test' # directories.Train     # should specify by user
    user_Output = '/array/ssd/msmajdi/experiments/Keras/vimp2_test' # directories.Results  # should specify by user

    directories.Input = checkInputDirectory( user_Input ,TrainParams.Nucleus.Name)
    directories.Output = checkOutputDirectory(user_Output , InputImageParams.SubExperiment_Number)
    del user_Input , user_Output

# --cropping mode
# 1 or mask:     cropping using the cropped mask acquired from rigid transformation
# 2 or thalamus: cropping using the cropped mask for plain size and Thalamus Prediction for slice numbers
# 3 or both:     cropping using the Thalamus prediction
Method = 2
class Cropping:
    Mode = True
    Method = whichCropMode(TrainParams.NucleusName, Method)  # it changes the mode to 1 if we're analyzing the Thalamus

del Method


class preprocess:
    Augment = Augment
    Cropping = Cropping
    Normalize = Normalize

del Augment, Cropping, Normalize
