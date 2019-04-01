

Model_Method =  'HCascade' #'FCN_2D' # HCascade' # 
Local_Flag = False

class SubExperiment:
    Index = 10
    Tag   = '' # 'WoFixedCrop'
    
class Experiments:
    Index = '7'
    Tag = 'cascadeV1'

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
    GPU_Index     = "7"
    Learning_Rate = 1e-3
    num_Layers    = 3
    NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
    nucleus_Index = [1]
    slicingDim    = [0,1,2]
    batch_size    = 80
    InputImage2Dvs3D = 2
    FirstLayer_FeatureMap_Num = 64
    verbose = 1
    Multiply_By_Thalmaus = True

    Initialize_FromThalamus   = False
    Initialize_FromOlderModel = False
    Initialize_From_3T = True

    save_Best_Epoch_Model = True

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

if Local_Flag: Experiments_Address = '/home/artinl/Documents/research'
else: Experiments_Address = '/array/ssd/msmajdi/experiments/keras'


#! Preprocessing
class preprocess:
    Mode = False
    BiasCorrection = False

class normalize:
    Mode = True
    Method = 'MinMax'

AugmentMode = False
Augment_LinearMode = True
Augment_Linear_Length = 6

class Augment_Rotation:
    Mode = True
    AngleMax = '7_6cnts' # '7' # 7_4cnts

class Augment_Shear:
    Mode = False
    ShearMax = 4

Augment_NonLinearMode = False

SaveReportMethod = 'pickle'
