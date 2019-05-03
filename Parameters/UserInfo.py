
Model_Method =  'mUnet' #''Cascade' # FCN_25D' #  HCascade' # 

# TypeExperiment == 1: # Main
# TypeExperiment == 2: # Transfer Learn ET
# TypeExperiment == 3: # SRI
# TypeExperiment == 4: # Predict ET from MS&Ctrl
# TypeExperiment == 5: # Train ET Initialized from 3T
# TypeExperiment == 6: # Train Main+ET
# TypeExperiment == 7: # Train Main+ET+SRI
# TypeExperiment == 8: # Train Main+SRI
# TypeExperiment == 9: # Train ET Initialized from Main+SRI
# TypeExperiment == 10: # Main + All Augments
# TypeExperiment == 11: # Main + Init from Thalamus
# TypeExperiment == 12: # Main + Init from 3T
TypeExperiment = 1

class CrossVal:
    Mode = True
    index = 'a'
    All_Indexes = ['a' , 'b']

class Experiments:
    Index , Tag = '3' , '' # '1' , '' # , 'cascadeV1'

DropoutValue = 0.3

class SubExperiment: 
    Index = 12
    Tag   = '_Main_InitFrom_SRI' # '_InitFrom_SRI_AllAugments' # _Main_PlustET_InitFrom_SRI  _Main_HighEpochs_InitFrom3T '_ET_InitFrom_Main_PlusSRI'  _Main_PlusSRI _Main_PlusET_PlusSRI _Main Generator_ '_SRI2' 'MainPlusET' # Cascade_FM20_7T'
    Mode_JustThis = False
  
class InitializeB:
    FromThalamus   = False
    FromOlderModel = False
    From_3T        = True
    # FromArbitrary = True
    # arbitraryInit = '_Main_PlusSRI'

class simulation:
    TestOnly      = True
    epochs        = 100
    GPU_Index     = "6"
    Learning_Rate = 1e-3
    num_Layers    = 3
    NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
    nucleus_Index = [1] # ,2,4]
    slicingDim    = [2,1,0]
    batch_size    = 100
    InputImage2Dvs3D = 2
    FirstLayer_FeatureMap_Num = 20
    verbose = 2
    Multiply_By_Thalmaus = False
    Multiply_By_Rest_For_AV = False

    Weighted_Class_Mode = False
    Initialize = InitializeB()
    save_Best_Epoch_Model = True
    Use_Coronal_Thalamus_InSagittal = True
    Use_TestCases_For_Validation = True
    ImClosePrediction = True
    

class dataGenerator:
    Mode = False
    NumSubjects_Per_batch = 5


class InputPadding:
    Automatic = True
    HardDimensions = [116,144,84]

if Experiments.Index == '8': 
    InputPadding.HardDimensions = [228,288,168]


mode_saveTrue_LoadFalse = True
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
    AngleMax = 7 # '7_6cnts' # '7' # 7_4cnts

class Augment_Shear:
    Mode = False
    ShearMax = 4

Augment_NonLinearMode = False

SaveReportMethod = 'pickle'

