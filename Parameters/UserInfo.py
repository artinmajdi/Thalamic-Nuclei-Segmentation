

Model_Method =   'Cascade' #'mUnet' #' FCN_25D' #  'HCascade' # 

# Main = 7T (Ctrl&MS)
# ET   = ET(7T + 3T)
# TypeExperiment == 1: # 3T      Init Randomly
# TypeExperiment == 2: # Main    Init from 3T
# TypeExperiment == 3: # ET      Init from Main
# TypeExperiment == 4: # ET Transfer Learn from Main
# TypeExperiment == 5: # ET Predicted from Main
# TypeExperiment == 6: # Main + 3T  Init Randomly
# TypeExperiment == 7: # ET      Init from Randomly
# TypeExperiment == 8 # Main + 3T  Init Th
TypeExperiment = 8

class CrossVal:
    Mode = True
    index = ['a']
    All_Indexes = ['a' , 'b' , 'c' , 'd']

class Experiments:
    Index , Tag = '3' , '' # '1' , '' # , 'cascadeV1'

DropoutValue = 0.3

class SubExperiment:
    Index = 11
    Tag   = '_Main_PlusSRI_InitFrom_Th' # _Main_Init_3T_AllAugs _ET_Init_Main_AllAugs _sE11_Cascade_FM20_DO0.3_Main_PlusSRI_InitFrom_Th_CV_a
    Mode_JustThis = False
  
class InitializeB:
    FromThalamus   = False
    FromOlderModel = False
    From_3T        = False
    From_7T        = False

class simulation:
    TestOnly      = False
    epochs        = 100
    GPU_Index     = "0,1"
    Learning_Rate = 1e-3
    num_Layers    = 3
    NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
    nucleus_Index = [1,2] # ,2,4]
    slicingDim    = [2] #[2,1,0]
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

