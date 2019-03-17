
readAugments = False

read3T_Mode = True
read3T_Tag = 'SRI'

Transfer_Learning_Mode = False
Transfer_Learning_Layers = [0,1]

Initialize_FromOlderModel = False


Model_Method = 'Cascade' # 'Hierarchical_Cascade' #
TestOnly = False
SubExperiment_Index = 445
mode_saveTrue_LoadFalse = True

lossFunctionIx = 5
havingBackGround_AsExtraDimension = True

class_weights = {0:1, 1:1e6}
dropout = 0.3
epochs = 15
GPU_Index = "4"
Learning_Rate = 1e-3
num_Layers = 3
NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
Experiments_Index = '7'
Experiments_Tag = 'cascadeV1'  # 'cropping' # initilization' # 'cascadeV1_3TforInit7T' #

nucleus_Index = [1]

slicingDim = [2]

# 1: ('SRI_3T', '/array/ssd/msmajdi/data/preProcessed/3T/SRI_3T'),
# 2: ('SRI_ReSliced', '/array/ssd/msmajdi/data/preProcessed/3T/SRI_ReSliced'),
# 3: ('croppingData', '/array/ssd/msmajdi/data/preProcessed/croppingData'),
# 4: ('All_7T', '/array/ssd/msmajdi/data/preProcessed/7T/All_DBD'),
# 5: ('20priors', '/array/ssd/msmajdi/data/preProcessed/7T/20priors'),
DatasetIx = 4


#! Training
batch_size = 200 # 100
Initialize_FromThalamus = False


InputPadding_Automatic = True
InputPadding_HardDimensions = 2 # [112,112,0]

#! GPU
# GPU_Index = 6


gapDilation = 5
#! Template Address
Tempalte_Image = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/origtemplate.nii.gz'
Tempalte_Mask = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/CropMaskV3.nii.gz'  # MyCrop_Template2_Gap20


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

# Experiments_Tag = '7T' # 'SRI' 'tmp' 'SRI_wLRAug' '7T' '7T_wLRAug'
SubExperiment_Tag = Model_Method

cropping_method = 'python' # 'ANTs' 'python'


#! Preprocessing
preprocessMode = True
BiasCorrection = False
Cropping = True
Normalize = True

#! this flag has two applications:
#    1. Called by dataset: to load the augmented data if available alongside dataset while creatting an experiment
#    2. Called by preprocess: to augment data inside train folder of the assigned experiment

AugmentMode = False
Augment_LinearMode = True


Augment_Rotation, Augment_Shear, Augment_Merge, Augment_Shift = [False]*4

# ! below is just for the sake of reading Augment data since we're not doing two augments at once at the moment
indAugment = 'Augment_Rotation'
if indAugment == 'Augment_Rotation': Augment_Rotation = True
elif indAugment == 'Augment_Shear':  Augment_Shear = True
elif indAugment == 'Augment_Shift':  Augment_Shift = True
elif indAugment == 'Augment_Merge':  Augment_Merge = True


Augment_ShiftMax = 10
Augment_ShearMax = 4
Augment_AngleMax = 7 # 15 #

Augment_NonLinearMode = False

#! save the report
#           'pickle'
#           'mat'
#           'json'
SaveReportMethod = 'pickle'
