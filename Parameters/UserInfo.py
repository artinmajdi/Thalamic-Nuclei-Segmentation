
# AllExperimentsList = {
#     1: dict(nucleus_Index = [8] , GPU_Index = 4 , lossFunctionIx = 3)
#     # 2: dict(nucleus_Index = [6] , GPU_Index = 6 , lossFunctionIx = 2),
#     # 3: dict(nucleus_Index = [6] , GPU_Index = 7 , lossFunctionIx = 3),
#     # 4: dict(nucleus_Index = [8] , GPU_Index = 5 , lossFunctionIx = 1),
#     # 5: dict(nucleus_Index = [8] , GPU_Index = 6 , lossFunctionIx = 2),
#     # 6: dict(nucleus_Index = [8] , GPU_Index = 7 , lossFunctionIx = 3),
# }



#! Nucleus Index
nucleus_Index = [1]


#! Training
num_Layers = 5
epochs = 30
batch_size = 40
Initialize_FromThalamus = True
Initialize_FromOlderModel = False

#! GPU
GPU_Index = 5



#! Template Address
Tempalte_Image = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/origtemplate.nii.gz'
Tempalte_Mask = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/MyCrop_Template2_Gap20.nii.gz'


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

#! Dataset
# DatasetIx =     1: 'SRI_3T'
#                 2: 'kaggleCompetition'
#                 3: 'fashionMnist'
#                 4: 'All_7T': 20priros + MS
#                 5: '20priros'
DatasetIx = 4
CreatingTheExperiment = True

#! Optimizer
#          1: 'Adam'
OptimizerIx = 1


#! Experiments Address
Experiments_Address = '/array/ssd/msmajdi/experiments/keras'
Experiments_Index = 2
Experiments_Tag = '7T' # 'SRI' 'tmp' 'SRI_wLRAug' '7T
SubExperiment_Index = 1
SubExperiment_Tag = '' 

#! cropping mode
#           1 or mask:     cropping using the cropped mask acquired from rigid transformation
#           2 or thalamus: cropping using the cropped mask for plain size and Thalamus Prediction for slice numbers
#           3 or both:     cropping using the Thalamus prediction
cropping_method = 2


#! Preprocessing
preprocessMode = False
BiasCorrection = False
Cropping = True
Normalize = True
TestOnly = False

#! this flag has two applications:
#    1. Called by dataset: to load the augmented data if available alongside dataset while creatting an experiment 
#    2. Called by preprocess: to augment data inside train folder of the assigned experiment
AugmentMode = True  
Augment_Rotation     = True
Augment_Shift        = False
Augment_NonRigidWarp = False

#if AugmentMode:
#    Experiments_Tag = 'SRI_wLRAug' # 'tmp' ''
#else:    
#    Experiments_Tag = 'SRI'


#! save the report
#           'pickle'
#           'mat'
#           'json'
SaveReportMethod = 'pickle'
