

Model_Method = 'Cascade' #'mUnet' # 'HCascade' # 'normal' #
architectureType = 'Res_Unet2' # 'U-Net4' #  'Res_Unet' # 'FCN_Unet_TL' # 'FCN_Unet' # ''FCN_Unet_TL' #  'SegNet_Unet' # 'SegNet' #  'FCN_Unet' # 'FCN'  #'FCN_with_SkipConnection' #  
gpu = "3"

local_flag = False
container_flag = False
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

TypeExperiment = 8


#! Preprocessing
class preprocesscs:
    def __init__(self):
        self.Mode = False
        self.BiasCorrection = False
        self.Cropping = False
        self.Reslicing = False

preprocess = preprocesscs()






permutation_Index = 0

multi_Class_Mode = True
readAugments_Mode = True
lossFunction_Index = 4

tag_temp = '' # _NEW' # _temp_fixed_BB
best_network_MPlanar = False
testOnly = False

fCN1_NLayers = 0
fCN2_NLayers = 0

class normalizeCs:
    def __init__(self):
            
        self.Mode = True
        self.Method = '1Std0Mean' #  'MinMax' #  'Both' # 
        self.per_Subject = True
        self.per_Dataset = False

normalize = normalizeCs()

class CrossValcs:
    def __init__(self):
        self.Mode = True
        self.index = ['a']
        self.All_Indexes = ['a' , 'b' , 'c' , 'd' , 'e' , 'f' , 'g' , 'h']

CrossVal = CrossValcs()

class Experimentscs:
    def __init__(self):
        self.Index , self.Tag = '7' , '' # '5_CSFn' , '' #'4' , '' #    '1' , '' # , 'cascadeV1'

Experiments = Experimentscs()

DropoutValue = 0.3

class SubExperimentcs:
    def __init__(self):
        self.Index = 12
        self.Tag   = '' # '_zeroWeightFor0257' _equal_weights' '_Main_PlusSRI_InitFrom_Th' # _Main_Init_3T_AllAugs _ET_Init_Main_AllAugs _sE11_Cascade_FM20_DO0.3_Main_PlusSRI_InitFrom_Th_CV_a
        self.Mode_JustThis = False

SubExperiment = SubExperimentcs()

class InitializeBcs:
    def __init__(self):
        self.FromThalamus   = False
        self.FromOlderModel = False
        self.From_3T        = False
        self.From_7T        = False
        self.From_CSFn      = False

InitializeB = InitializeBcs()

class upsamplecs:
    def __init__(self):
        self.Mode = True
        self.Scale = 1

upsample = upsamplecs()

class simulationcs:
    def __init__(self):
        self.TestOnly      = testOnly
        self.epochs        = 300
        self.GPU_Index     = gpu
        self.Learning_Rate = 1e-3
        self.num_Layers    = 3 
        self.FCN1_NLayers  = fCN1_NLayers
        self.FCN2_NLayers  = fCN2_NLayers
        self.nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
        self.slicingDim    = [2 ,1 ,0]
        self.batch_size    = 10
        self.InputImage2Dvs3D = 2
        self.FirstLayer_FeatureMap_Num = 20
        self.FCN_FeatureMaps = 30
        self.verbose = 2
        self.Multiply_By_Thalmaus = False
        self.Multiply_By_Rest_For_AV = False

        self.Weighted_Class_Mode = True
        self.Initialize = InitializeB
        self.save_Best_Epoch_Model = True
        self.Use_Coronal_Thalamus_InSagittal = True
        self.Use_TestCases_For_Validation = True
        self.ImClosePrediction =  True # False #
        self.Multi_Class_Mode = multi_Class_Mode
        self.LR_Scheduler = False
        self.ReadAugments_Mode = readAugments_Mode
    
simulation = simulationcs()

class dataGeneratorcs:
    def __init__(self):
        self.Mode = False
        self.NumSubjects_Per_batch = 5

dataGenerator = dataGeneratorcs()


class InputPaddingcs:
    def __init__(self):
        self.Automatic = True
        self.HardDimensions = [116,144,84]

InputPadding = InputPaddingcs()

if Experiments.Index == '8': 
    InputPadding.HardDimensions = [228,288,168]


mode_saveTrue_LoadFalse = True
havingBackGround_AsExtraDimension = True

gapDilation = 5



code_address = '/array/ssd/msmajdi/code/thalamus/keras/'
class Templatecs:
    def __init__(self):
        self.Image = code_address + 'general/RigidRegistration' + '/origtemplate.nii.gz'
        self.Mask = code_address + 'general/RigidRegistration' + '/CropMaskV3.nii.gz'  # MyCrop_Template2_Gap20
        # self.Mask_2AV = '/array/ssd/msmajdi/code/thalamus/keras/general/RigidRegistration' + '/CropMask_AV.nii.gz' 
        self.Address = code_address + 'general/RigidRegistration/'

Template = Templatecs()


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


if local_flag:
    Experiments_Address = '/media/artin/SSD/RESEARCH/PhD/Experiments'
    Template.Image = '/media/artin/SSD/RESEARCH/PhD/code/general/RigidRegistration/origtemplate.nii.gz'
    Template.Mask = '/media/artin/SSD/RESEARCH/PhD/code/general/RigidRegistration/CropMaskV3.nii.gz'
    Template.Address = '/media/artin/SSD/RESEARCH/PhD/code/general/RigidRegistration/'

if container_flag:
    Experiments_Address = '/Experiments'
    Template.Image = '/code/general/RigidRegistration/origtemplate.nii.gz'
    Template.Mask = '/code/general/RigidRegistration/CropMaskV3.nii.gz'
    Template.Address = '/code/general/RigidRegistration/'


AugmentMode = False
Augment_LinearMode = True
Augment_Linear_Length = 3

class Augment_Rotationcs:
    def __init__(self):
        self.Mode = True
        self.AngleMax = 7 # '7_6cnts' # '7' # 7_4cnts

Augment_Rotation = Augment_Rotationcs()

class Augment_Shearcs:
    def __init__(self):
        self.Mode = False
        self.ShearMax = 4

Augment_Shear = Augment_Shearcs()



Augment_NonLinearMode = False

SaveReportMethod = 'pickle'

