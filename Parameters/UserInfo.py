

Model_Method =  'Cascade' #'FCN_2D' # HCascade' # 
Local_Flag = False
mode3T_7T = '7T'

class SubExperiment:
    Index = 11
    Tag   = ''
    
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
    Automatic = True
    HardDimensions = [1,1,1] # [116,144,84]


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
    epochs        = 40
    GPU_Index     = "3"
    Learning_Rate = 1e-3
    num_Layers    = 3
    NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
    nucleus_Index = [2]
    slicingDim    = [2,1] # [0,1,2]
    batch_size    = 100
    InputImage2Dvs3D = 2
    FirstLayer_FeatureMap_Num = 20
    verbose = 2
    Multiply_By_Thalmaus = False

    Initialize_FromThalamus   = True
    Initialize_FromOlderModel = True
    Initialize_From_3T = True
    Weighted_Class_Mode = False

    save_Best_Epoch_Model = True
    Use_Coronal_Thalamus_InSagittal = True
    Use_TestCases_For_Validation = True
    ImClosePrediction = True

if mode3T_7T == '3T':
    simulation.Initialize_From_3T = False
    ReadTrain.SRI = True
    ReadTrain.Main = False
    SubExperiment.Index = 8

    
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
