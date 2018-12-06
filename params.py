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
    Organ = 'THALAMUS'
    Name , FullIndexes = NucleiSelection( Index[0] , Organ)

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
    Mode = True
    LinearAugmentLength = 1  # number
    NonLinearAugmentLength = 1
    Rotation = True
    Shift = False
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
    Mode = True

class Debug:
    doDebug = True
    PProcessExist = False  # rename it to preprocess exist
    justForNow = True # it checks the intermediate steps and if it existed don't reproduce it

class preprocess:
    Mode = True
    TestOnly = False
    Debug = Debug
    Augment = Augment
    Cropping = Cropping
    Normalize = Normalize
    BiasCorrection = BiasCorrection

del Augment, Cropping, Normalize, template, reference, nucleus, experiment, machine, model, image, hardParams, method , Debug , BiasCorrection

