
readAugmentsMode = True

class ReadMain:
    Mode = True
class Read3T:
    Mode = False
    Tag = 'SRI'

class InputPadding:
    Automatic = False
    HardDimensions = [116,144,84]

class Transfer_Learning:
    Mode = False
    FrozenLayers = [0]
    Stage = 0

class simulation:
    TestOnly      = False
    epochs        = 20
    GPU_Index     = "4"
    Learning_Rate = 1e-3
    num_Layers    = 3
    NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
    nucleus_Index = [1]
    slicingDim    = [2]
    batch_size    = 100

    Initialize_FromThalamus   = False
    Initialize_FromOlderModel = False



Model_Method = 'Cascade' # 'Hierarchical_Cascade' # 

class SubExperiment:
    Index = 6
    Tag   = Model_Method

mode_saveTrue_LoadFalse = True

class dropout:
    Mode = True
    Value = 0.3
    
DatasetIx = 4

lossFunctionIx = 5
havingBackGround_AsExtraDimension = True



class Experiments:
    Index = '7'
    Tag = 'cascadeV1' 

gapDilation = 5

class Template:
    Image = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/origtemplate.nii.gz'
    Mask = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/CropMaskV3.nii.gz'  # MyCrop_Template2_Gap20


#! MultiClass
MultiClass_mode = False


#! metric function
#          1: 'Dice'
#          2: 'Accuracy'
#          3: 'Dice & Accuracy'
MetricIx = 3
OptimizerIx = 1
# Learning_Rate = 1e-3
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
    AngleMax = 7 # 15

class Augment_Shear:
    Mode = False
    ShearMax = 4   
    
Augment_NonLinearMode = False

SaveReportMethod = 'pickle'
