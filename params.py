import os
from smallFuncs import NucleiSelection , whichCropMode , fixDirectoryLastDashSign , augmentLengthChecker , InputNames , checkInputDirectory , funcExpDirectories

#  ---------------------- model Params ----------------------

class template:
    Image = '/array/ssd/msmajdi/code/RigidRegistration' + '/origtemplate.nii.gz'
    Mask = '/array/ssd/msmajdi/code/RigidRegistration' + '/MyCrop_Template2_Gap20.nii.gz'   

class model:
    ArchitectureType = 'U-Net'
    NumberOfLayers = 3
    Optimizer = 'Adam'

class machine:
    WhichMachine = 'server'
    GPU_Index = '1'

class image:
    SlicingDirection = 'axial'.lower()
    SaveMode = 'nifti'.lower()

class nucleus:
    Index = [1]
    Name = NucleiSelection( Index[0] )

class hardParams:
    Model    = model
    Template = template
    Machine  = machine
    Image    = image

class experiment:
    Tag = 'firstExperiment'
    Experiment_Index    = 1
    SubExperiment_Index = 1
    Address = '/array/ssd/msmajdi/experiments/Keras'
    Nucleus = nucleus
    HardParams = hardParams

if 0: # user defined address
    experiment.Address =  '/array/ssd/msmajdi/experiments/Keras/Experiment1/'

directories = funcExpDirectories(experiment)

class reference:
    Name = ''
    Address = ''

class Augment:
    Mode = False
    LinearAugmentLength = 2  # number
    NonLinearAugmentLength = 3
    Rotation = True
    Shift = True
    NonRigidWarp = False

Augment = augmentLengthChecker(Augment)

class Normalize:
    Mode = True
    Method = 'MinMax'.lower()



# --cropping mode
# 1 or mask:     cropping using the cropped mask acquired from rigid transformation
# 2 or thalamus: cropping using the cropped mask for plain size and Thalamus Prediction for slice numbers
# 3 or both:     cropping using the Thalamus prediction
method = 2
class Cropping:
    Mode = True
    Method = whichCropMode(directories.Experiment.Nucleus.Name, method)  # it changes the mode to 1 if we're analyzing the Thalamus

class BiasCorrection:
    Mode = False

class Debug:
    Mode = True

class preprocess:
    Mode = True
    Debug = Debug
    Augment = Augment
    Cropping = Cropping
    Normalize = Normalize
    BiasCorrection = BiasCorrection

del Augment, Cropping, Normalize, template, reference, nucleus, experiment, machine, model, image, hardParams, method

