import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs.smallFuncs import NucleiSelection, whichCropMode, fixDirectoryLastDashSign, augmentLengthChecker, InputNames, checkInputDirectory, funcExpDirectories, mkDir
from keras import losses, optimizers, metrics
import tensorflow as tf
import numpy as np
#  ---------------------- model Params ----------------------

def Dice_Calculator(msk1,msk2):
    return tf.reduce_sum(tf.multiply(msk1,msk2))*2/( tf.reduce_sum(msk1) + tf.reduce_sum(msk2) + tf.keras.backend.epsilon())

class template:
    Image = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/origtemplate.nii.gz'
    Mask = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/MyCrop_Template2_Gap20.nii.gz'

class dropout:
    Mode = True
    Value = 0.2

class kernel_size:
    conv = (3,3)
    convTranspose = (2,2)
    output = (1,1)

class activation:
    layers = 'relu'
    output = 'sigmoid'

class convLayer:
    # strides = (1,1)
    Kernel_size = kernel_size
    padding = 'SAME' # valid

class maxPooling:
    strides = (2,2)
    pool_size = (2,2)

class model:
    architectureType = 'U-Net' # 'U-Net' # 'MLP' #
    epochs = 2
    batch_size = 50
    loss = losses.categorical_crossentropy   # binary_crossentropy  sparse_categorical_crossentropy  cosine_proximity
    metrics = ['acc',Dice_Calculator]
    optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # adamax Nadam Adadelta Adagrad
    num_Layers = 3
    InputDimensions = ''
    batchNormalization = False # True
    ConvLayer = convLayer
    MaxPooling = maxPooling
    Dropout = dropout
    Activitation = activation
    showHistory = True
    LabelMaxValue = 1


class machine:
    WhichMachine = 'server'
    GPU_Index = str(5)

class image:
    SlicingDirection = 'axial'.lower()
    SaveMode = 'nifti'.lower()

class nucleus:
    Index = [1]
    Organ = 'THALAMUS' # 'Hippocampus
    name , FullIndexes = NucleiSelection( Index[0] , Organ)

class hardParams:
    Model    = model
    Template = template
    Machine  = machine
    Image    = image

class experiment:
    index = 1
    tag = 'tmp'
    name = ''
    address = ''
experiment.name = 'exp' + str(experiment.index) + '_' + experiment.tag if experiment.tag else 'subExp' + str(experiment.index)

class subExperiment:
    index = 1
    tag = 'LR00_' + nucleus.name
    name = ''
subExperiment.name = 'subExp' + str(subExperiment.index) + '_' + subExperiment.tag if subExperiment.tag else 'subExp' + str(subExperiment.index)

class validation:
    percentage = 0.1
    fromKeras = True

class test:
    mode = 'percentage' # 'names'
    percentage = 0.3
    subjects = ''

# TODO IMPORT TEST SUBJECTS NAMES AS A LIST
if 'names' in test.mode: # import test.subjects
    test.subjects = list([''])

class dataset:
    name = 'SRI_3T' #'kaggleCompetition' #  'fashionMnist' #
    address = ''
    Validation = validation
    Test = test
    onlySubjectsWvimp = True

if 'SRI_3T' in dataset.name:
    dataset.address = '/array/ssd/msmajdi/data/preProcessed/SRI_3T'
elif 'kaggleCompetition' in dataset.name:
    dataset.address = '/array/ssd/msmajdi/data/originals/KaggleCompetition/train'
elif 'fashionMnist' in dataset.name:
    dataset.address = 'intrinsic'

class WhichExperiment:
    Experiment    = experiment
    SubExperiment = subExperiment
    address = mkDir('/array/ssd/msmajdi/experiments/keras')
    Nucleus = nucleus
    HardParams = hardParams
    Dataset = dataset
WhichExperiment.Experiment.address = mkDir(WhichExperiment.address + '/' + WhichExperiment.Experiment.name)

directories = funcExpDirectories(WhichExperiment)

class reference:
    name = ''
    address = ''

class Augment:
    Mode = False
    LinearAugmentLength = 1  # number
    NonLinearAugmentLength = 1
    Rotation = True
    Shift = False
    NonRigidWarp = False

Augment = augmentLengthChecker(Augment)

class Normalize:
    Mode = True
    Method = 'MinMax'


# --cropping mode
# 1 or mask:     cropping using the cropped mask acquired from rigid transformation
# 2 or thalamus: cropping using the cropped mask for plain size and Thalamus Prediction for slice numbers
# 3 or both:     cropping using the Thalamus prediction
method = 2
class Cropping:
    Mode = True
    Method = whichCropMode(WhichExperiment.Nucleus.name, method)  # it changes the mode to 1 if we're analyzing the Thalamus

class BiasCorrection:
    Mode = False

# TODO fix the justfornow
class Debug:
    doDebug = True
    PProcessExist = False  # rename it to preprocess exist
    justForNow = True # it checks the intermediate steps and if it existed don't reproduce it

class preprocess:
    Mode = ''
    CreatingTheExperiment = ''
    TestOnly = False
    Debug = Debug
    Augment = Augment
    Cropping = Cropping
    Normalize = Normalize
    BiasCorrection = BiasCorrection

del subExperiment, test, dataset, sys, Augment, Cropping, Normalize, template, reference, nucleus, experiment, machine, model, image, hardParams, method , Debug , BiasCorrection , validation, activation, maxPooling, dropout, convLayer , kernel_size , os
