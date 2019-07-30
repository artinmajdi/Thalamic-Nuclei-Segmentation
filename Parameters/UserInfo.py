

Model_Method = 'Cascade' #'mUnet' # 'HCascade' # 'normal' #
architectureType = 'U-Net4' #  'Res_Unet' # 'FCN_Unet_TL' # 'FCN_Unet' # ''FCN_Unet_TL' #  'SegNet_Unet' # 'SegNet' #  'FCN_Unet' # 'FCN'  #'FCN_with_SkipConnection' #  
gpu = "4"

# TypeExperiment == 1: # 3T      Init Rn

# TypeExperiment == 2:  # Main        Init 3T
# TypeExperiment == 14: # Main        Init Rn
# TypeExperiment == 15: # Main+ET     Init 3T
# TypeExperiment == 16: # 3T+Main+ET  Init 3T

# TypeExperiment == 3   # Main + 3T  Init 3T

# TypeExperiment == 4: # ET      Init Main
# TypeExperiment == 5: # ET Transfer Learn from Main

# TypeExperiment == 6  # CSFn1  Init Main

# TypeExperiment == 7  # CSFn2  Init CSFn1
# TypeExperiment == 8  # CSFn2  Init Main
# TypeExperiment == 9  # CSFn2  Transfer Learn from CSFn1
# TypeExperiment == 10 # CSFn2  Transfer Learn from Main  old structure wher it was fully initializing from a network
# TypeExperiment == 11 # CSFn2  Transfer Learn from Main with new structure wher it only takes the wweights for part of the network
# TypeExperiment == 13  # CSFn1 + CSFn2 Init Main

TypeExperiment = 4

multi_Class_Mode = True
readAugments_Mode = True
lossFunction_Index = 3

tag_temp = '' # _NEW' # _temp_fixed_BB
testOnly = False

fCN1_NLayers = 3
fCN2_NLayers = 1

class normalize:
    Mode = True
    Method = '1Std0Mean' #  'MinMax' #  'Both' # 
    per_Subject = True
    per_Dataset = False



class CrossVal:
    Mode = True
    index = ['a']
    All_Indexes = ['a' , 'b' , 'c' , 'd']

class Experiments:
    Index , Tag = '6' , '' # '5_CSFn' , '' #'4' , '' #    '1' , '' # , 'cascadeV1'


DropoutValue = 0.3

class SubExperiment:
    Index = 12
    Tag   = '' # '_zeroWeightFor0257' _equal_weights' '_Main_PlusSRI_InitFrom_Th' # _Main_Init_3T_AllAugs _ET_Init_Main_AllAugs _sE11_Cascade_FM20_DO0.3_Main_PlusSRI_InitFrom_Th_CV_a
    Mode_JustThis = False
  
class InitializeB:
    FromThalamus   = False
    FromOlderModel = False
    From_3T        = False
    From_7T        = False
    From_CSFn      = False

class upsample:
    Mode = True
    Scale = 1

class simulation:
    TestOnly      = testOnly
    epochs        = 300
    GPU_Index     = gpu
    Learning_Rate = 1e-3
    num_Layers    = 3 
    FCN1_NLayers  = fCN1_NLayers
    FCN2_NLayers  = fCN2_NLayers
    nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
    slicingDim    = [2 ,1 ,0]
    batch_size    = 100
    InputImage2Dvs3D = 2
    FirstLayer_FeatureMap_Num = 20
    FCN_FeatureMaps = 30
    verbose = 2
    Multiply_By_Thalmaus = False
    Multiply_By_Rest_For_AV = False

    Weighted_Class_Mode = True
    Initialize = InitializeB()
    save_Best_Epoch_Model = True
    Use_Coronal_Thalamus_InSagittal = True
    Use_TestCases_For_Validation = True
    ImClosePrediction =  True # False #
    Multi_Class_Mode = multi_Class_Mode
    LR_Scheduler = False
    ReadAugments_Mode = readAugments_Mode
    

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
    Image = '/array/ssd/msmajdi/code/thalamus/keras/general/RigidRegistration' + '/origtemplate.nii.gz'
    Mask = '/array/ssd/msmajdi/code/thalamus/keras/general/RigidRegistration' + '/CropMaskV3.nii.gz'  # MyCrop_Template2_Gap20
    # Mask_2AV = '/array/ssd/msmajdi/code/thalamus/keras/general/RigidRegistration' + '/CropMask_AV.nii.gz' 
    Address = '/array/ssd/msmajdi/code/thalamus/keras/general/RigidRegistration/'


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


AugmentMode = False
Augment_LinearMode = True
Augment_Linear_Length = 2

class Augment_Rotation:
    Mode = True
    AngleMax = 7 # '7_6cnts' # '7' # 7_4cnts

class Augment_Shear:
    Mode = False
    ShearMax = 4

Augment_NonLinearMode = False

SaveReportMethod = 'pickle'

