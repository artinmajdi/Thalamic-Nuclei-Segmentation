

Model_Method =  'HCascade' # 'Cascade' #

class SubExperiment:
    Index = 7
    Tag   = Model_Method

class readAugments:
    Mode = True
    Tag = ''

class ReadTrain:
    SRI = False
    ET = False
    Main = True
    ReadAugments = readAugments()

class InputPadding:
    Automatic = False
    HardDimensions = [116,144,84]

class Transfer_Learning:
    Mode = False
    FrozenLayers = [0]
    Stage = 0

class simulation:
    TestOnly      = False
    epochs        = 100
    GPU_Index     = "7"
    Learning_Rate = 1e-4
    num_Layers    = 3
    NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
    nucleus_Index = [2]
    slicingDim    = [0]
    batch_size    = 100

    Initialize_FromThalamus   = False
    Initialize_FromOlderModel = False



mode_saveTrue_LoadFalse = True
DropoutValue = 0.3
havingBackGround_AsExtraDimension = True

class Experiments:
    Index = '7'
    Tag = 'cascadeV1'

gapDilation = 5

class Template:
    Image = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/origtemplate.nii.gz'
    Mask = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/CropMaskV3.nii.gz'  # MyCrop_Template2_Gap20


#! metric function
#          1: 'Dice'
#          2: 'Accuracy'
#          3: 'Dice & Accuracy'
MetricIx = 3
OptimizerIx = 1
Learning_Rate = 1e-3
Experiments_Address = '/array/ssd/msmajdi/experiments/keras'


class cropping:
    Mode = True
    method = 'python' # 'ANTs' 'python'


#! Preprocessing
class preprocess:
    Mode = True
    BiasCorrection = False
    Cropping = cropping()
    Normalize = True


AugmentMode = False
Augment_LinearMode = True


class Augment_Rotation:
    Mode = True
    AngleMax = '7_4cnts' # 15

class Augment_Shear:
    Mode = False
    ShearMax = 4

Augment_NonLinearMode = False

SaveReportMethod = 'pickle'
