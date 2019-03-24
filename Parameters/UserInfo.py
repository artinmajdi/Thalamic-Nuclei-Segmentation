

Model_Method =  'Cascade' #'HCascade' # 

class SubExperiment:
    Index = 8
    Tag   = Model_Method
    
class Experiments:
    Index = '7'
    Tag = 'cascadeV1'

class readAugments:
    Mode = True
    Tag = ''

class ReadTrain:
    SRI = True
    ET = False
    Main = False
    ReadAugments = readAugments()

class InputPadding:
    Automatic = False
    HardDimensions = [116,144,84]

# sd0:  [288, 168, 228]
# sd1:  [168, 228, 288]
# sd2:  [228, 288, 168]
if Experiments.Index == '8': InputPadding.HardDimensions = [228,288,168]

class Transfer_Learning:
    Mode = False
    FrozenLayers = [0]
    Stage = 0

class simulation:
    TestOnly      = False
    epochs        = 100
    GPU_Index     = "1"
    Learning_Rate = 1e-3
    num_Layers    = 3
    NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
    nucleus_Index = [1,2,8]
    slicingDim    = [2]
    batch_size    = 50
    InputImage2Dvs3D = 2
    FirstLayer_FeatureMap_Num = 64

    Initialize_FromThalamus   = False
    Initialize_FromOlderModel = False



mode_saveTrue_LoadFalse = True
DropoutValue = 0.3
havingBackGround_AsExtraDimension = True

gapDilation = 5

class Template:
    Image = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/origtemplate.nii.gz'
    Mask = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/CropMaskV3.nii.gz'  # MyCrop_Template2_Gap20


#! metric function
#          1: 'Dice'
#          2: 'Accuracy'
#          3: 'Dice & Accuracy'
MetricIx = 3
Learning_Rate = 1e-3
Experiments_Address = '/array/ssd/msmajdi/experiments/keras'


#! Preprocessing
class preprocess:
    Mode = False
    BiasCorrection = False

class normalize:
    Mode = True
    Method = 'MinMax'

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
