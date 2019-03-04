
epochs = 30
GPU_Index = 6
Learning_Rate = 1e-3
num_Layers = 3

NormalizaeMethod = 'MinMax' #'1Std0Mean' # 
readAugments = False


TestOnly = False
Experiments_Index = '7' # 'cropping' # 7_croppingNetwork' # 
Experiments_Tag = 'cascadeV1'  # 'cascadeV1_3TforInit7T' # 

nucleus_Index = [1,2,8,10]

slicingDim = 2

# 1: ('SRI_3T', '/array/ssd/msmajdi/data/preProcessed/3T/SRI_3T'),
# 2: ('SRI_ReSliced', '/array/ssd/msmajdi/data/preProcessed/3T/SRI_ReSliced'),
# 3: ('croppingData', '/array/ssd/msmajdi/data/preProcessed/croppingData'),
# 4: ('All_7T', '/array/ssd/msmajdi/data/preProcessed/7T/All_DBD'),
# 5: ('20priors', '/array/ssd/msmajdi/data/preProcessed/7T/20priors'),
DatasetIx = 4
SubExperiment_Index = 2

#! Training
batch_size = 40
Initialize_FromThalamus = False
Initialize_FromOlderModel = False

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

#! loss function
# lossFunction=   1: 'dice'
#                 2: 'binary Cross Enropy'
#                 3: 'Both'
lossFunctionIx = 3

# orderDim =       2: [0,1,2]
# orderDim =       1: [2,0,1]
# orderDim =       0: [1,2,0]


#! Optimizer
#          1: 'Adam'
OptimizerIx = 1
# Learning_Rate = 1e-3

#! Experiments Address
Experiments_Address = '/array/ssd/msmajdi/experiments/keras'

# Experiments_Tag = '7T' # 'SRI' 'tmp' 'SRI_wLRAug' '7T' '7T_wLRAug'



SubExperiment_Tag = NormalizaeMethod 
if readAugments: SubExperiment_Tag += '_wAug'

#! cropping mode
#           'ANTs'
#           'python'
cropping_method = 'python' # 'ANTs' 'python'


#! Preprocessing
preprocessMode = False
BiasCorrection = False
Cropping = True
Normalize = True

#! this flag has two applications:
#    1. Called by dataset: to load the augmented data if available alongside dataset while creatting an experiment
#    2. Called by preprocess: to augment data inside train folder of the assigned experiment

AugmentMode = False
Augment_LinearMode = True

Augment_Rotation     = True
Augment_AngleMax = 7

Augment_Shift        = False
Augment_ShiftMax = 10

Augment_NonLinearMode = False

#! save the report
#           'pickle'
#           'mat'
#           'json'
SaveReportMethod = 'pickle'
