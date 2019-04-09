
Model_Method =  'HCascade' #'FCN_25D' #  HCascade' # 

# TypeExperiment == 1: # Main
# TypeExperiment == 2: # Transfer Learn ET
# TypeExperiment == 3: # SRI
# TypeExperiment == 4: # Predict ET from MS&Ctrl
# TypeExperiment == 5: # Train ET Initialized from 3T
# TypeExperiment == 6: # Train Main+ET
TypeExperiment = 6

class SubExperiment: 
        Index = 11
        Tag   = 'MainPlusET' # Cascade_FM20_7T'
        Mode_JustThis = False

class InitializeB:
    FromThalamus   = True
    FromOlderModel = True
    From_3T        = True

class simulation:
    TestOnly      = True
    epochs        = 100
    GPU_Index     = "5,6"
    Learning_Rate = 1e-3
    num_Layers    = 3
    NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
    nucleus_Index = [1] # ,2,4]
    slicingDim    = [2,1,0] # [0,1,2]
    batch_size    = 100
    InputImage2Dvs3D = 2
    FirstLayer_FeatureMap_Num = 20
    verbose = 2
    Multiply_By_Thalmaus = False

    Weighted_Class_Mode = False
    Initialize = InitializeB()
    save_Best_Epoch_Model = True
    Use_Coronal_Thalamus_InSagittal = True
    Use_TestCases_For_Validation = True
    ImClosePrediction = True



    
class InputPadding:
    Automatic = True
    HardDimensions = [1,1,1] # [116,144,84]

class Experiments:
    Index , Tag = '1' , '' # , 'cascadeV1'

if Experiments.Index == '8': InputPadding.HardDimensions = [228,288,168]


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

# if Local_Flag: 
#     Experiments_Address = '/home/artinl/Documents/research'
# else: 
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
Augment_Linear_Length = 6

class Augment_Rotation:
    Mode = True
    AngleMax = '7_6cnts' # '7' # 7_4cnts

class Augment_Shear:
    Mode = False
    ShearMax = 4

Augment_NonLinearMode = False

SaveReportMethod = 'pickle'

