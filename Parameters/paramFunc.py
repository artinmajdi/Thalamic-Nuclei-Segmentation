import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modelFuncs import LossFunction, Metrics, Optimizers
from Parameters import Classes
from otherFuncs import smallFuncs, datasets
from copy import deepcopy
import pandas as pd



class paramsA:
    WhichExperiment = Classes.WhichExperiment
    preprocess      = Classes.preprocess
    Augment         = Classes.Augment
    directories     = ''
    UserInfo        = ''
             

def Run(UserInfo):

    params = deepcopy(paramsA)
    # UserInfo = deepcopy(UserInfoB)

    WhichExperiment = deepcopy(params.WhichExperiment)
    preprocess      = deepcopy(params.preprocess)
    Augment         = deepcopy(params.Augment)

    WhichExperiment.address = smallFuncs.mkDir(UserInfo['Experiments_Address'])

    WhichExperiment.HardParams.Template.Image = UserInfo['Tempalte_Image']
    WhichExperiment.HardParams.Template.Mask  = UserInfo['Tempalte_Mask']
    WhichExperiment.HardParams.Model.MultiClass.mode = UserInfo['MultiClass_mode']
    WhichExperiment.HardParams.Model.loss, _      = LossFunction.LossInfo(UserInfo['lossFunctionIx'])
    WhichExperiment.HardParams.Model.metrics, _   = Metrics.MetricInfo(UserInfo['MetricIx'])
    WhichExperiment.HardParams.Model.optimizer, _ = Optimizers.OptimizerInfo(UserInfo['OptimizerIx'], UserInfo['Learning_Rate'])
    WhichExperiment.HardParams.Model.num_Layers   = UserInfo['num_Layers']
    WhichExperiment.HardParams.Model.batch_size   = UserInfo['batch_size']
    WhichExperiment.HardParams.Model.epochs       = UserInfo['epochs']
    WhichExperiment.HardParams.Model.InitializeFromThalamus = UserInfo['Initialize_FromThalamus']
    WhichExperiment.HardParams.Model.InitializeFromOlderModel = UserInfo['Initialize_FromOlderModel']
    WhichExperiment.HardParams.Machine.GPU_Index = str(UserInfo['GPU_Index'])

    if WhichExperiment.HardParams.Model.InitializeFromThalamus and WhichExperiment.HardParams.Model.InitializeFromOlderModel:
        print('WARNING:   initilization can only happen from one source')
        WhichExperiment.HardParams.Model.InitializeFromThalamus = False
        WhichExperiment.HardParams.Model.InitializeFromOlderModel = False


    WhichExperiment.Dataset.name, WhichExperiment.Dataset.address = datasets.DatasetsInfo(UserInfo['DatasetIx'])

    # orderDim =       2: [0,1,2]
    # orderDim =       1: [2,0,1]
    # orderDim =       0: [1,2,0]

    WhichExperiment.Dataset.slicingInfo.slicingDim = UserInfo['slicingDim']
    if UserInfo['slicingDim'] == 0:
        WhichExperiment.Dataset.slicingInfo.slicingOrder         = [1,2,0]
        WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse = [2,0,1]
    elif UserInfo['slicingDim'] == 1:
        WhichExperiment.Dataset.slicingInfo.slicingOrder         = [2,0,1]
        WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse = [1,2,0]
    else:
        WhichExperiment.Dataset.slicingInfo.slicingOrder         = [0,1,2]
        WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse = [0,1,2]

    WhichExperiment.SubExperiment.index = UserInfo['SubExperiment_Index']
    WhichExperiment.Experiment.index = UserInfo['Experiments_Index']

    if UserInfo['DatasetIx'] == 4:
        Experiments_Tag = '7T'
    elif UserInfo['DatasetIx'] == 1:
        Experiments_Tag = 'SRI'
    elif UserInfo['DatasetIx'] == 2:
        Experiments_Tag = 'Cropping'

    if UserInfo['AugmentMode']:  
        tagEx = ''
        if UserInfo['Augment_LinearMode']:
            if UserInfo['Augment_Rotation']: tagEx = tagEx + 'wLR'  + str(UserInfo['Augment_AngleMax'])
            if UserInfo['Augment_Shift']:    tagEx = tagEx + 'wLSh' + str(UserInfo['Augment_ShiftMax'])

        if UserInfo['Augment_NonLinearMode']: tagEx = tagEx + 'wNL' 

        if tagEx: Experiments_Tag = Experiments_Tag + '_' + tagEx + 'Aug'
            
        # Experiments_Tag = Experiments_Tag + '_wLRAug'

    WhichExperiment.Experiment.tag = Experiments_Tag   # UserInfo['Experiments_Tag']
    WhichExperiment.Experiment.name = 'exp' + str(UserInfo['Experiments_Index']) + '_' + WhichExperiment.Experiment.tag if WhichExperiment.Experiment.tag else 'exp' + str(WhichExperiment.Experiment.index)
    WhichExperiment.Experiment.address = smallFuncs.mkDir(WhichExperiment.address + '/' + WhichExperiment.Experiment.name)
    _, B = LossFunction.LossInfo(UserInfo['lossFunctionIx'])

    WhichExperiment.SubExperiment.tag = UserInfo['SubExperiment_Tag'] + B + '_sd' + str(UserInfo['slicingDim']) if int(UserInfo['slicingDim']) != 2 else UserInfo['SubExperiment_Tag'] + B
    # WhichExperiment.SubExperiment.tag = UserInfo['SubExperiment_Tag'] + 'lr' + str(UserInfo['Learning_Rate'])  + '_nl' + str(UserInfo['num_Layers']) 

    if 'U-Net' in WhichExperiment.HardParams.Model.architectureType:
        # WhichExperiment.SubExperiment.name = 'subExp' + str(WhichExperiment.SubExperiment.index) + '_' + WhichExperiment.SubExperiment.tag + WhichExperiment.Nucleus.name if WhichExperiment.SubExperiment.tag else 'subExp' + str(WhichExperiment.SubExperiment.index) + '_' + WhichExperiment.Nucleus.name
        AAA = 'subExp' + str(WhichExperiment.SubExperiment.index) + '_' + WhichExperiment.SubExperiment.tag if WhichExperiment.SubExperiment.tag else 'subExp' + str(WhichExperiment.SubExperiment.index)
        WhichExperiment.SubExperiment.name = AAA   + '_nl' + str(UserInfo['num_Layers'])  # + '5' #
    else:
        WhichExperiment.SubExperiment.name = 'subExp' + str(WhichExperiment.SubExperiment.index)

        
    # WhichExperiment.SubExperiment.name_thalamus = 'subExp' + str(WhichExperiment.SubExperiment.index) + '_' + WhichExperiment.SubExperiment.tag if WhichExperiment.SubExperiment.tag else 'subExp' + str(WhichExperiment.SubExperiment.index)


    # TODO I need to fix this to count for multiple nuclei
    WhichExperiment.Nucleus.Index = UserInfo['nucleus_Index'] if isinstance(UserInfo['nucleus_Index'],list) else [UserInfo['nucleus_Index']]
    
    WhichExperiment.Nucleus.name_Thalamus, WhichExperiment.Nucleus.FullIndexes, _ = smallFuncs.NucleiSelection( 1 , WhichExperiment.Nucleus.Organ)
    if len(WhichExperiment.Nucleus.Index) == 1 or not WhichExperiment.HardParams.Model.MultiClass.mode:
        WhichExperiment.Nucleus.name , _, _ = smallFuncs.NucleiSelection( WhichExperiment.Nucleus.Index[0] , WhichExperiment.Nucleus.Organ)
    else:
        WhichExperiment.Nucleus.name = ('MultiClass_' + str(WhichExperiment.Nucleus.Index)).replace(', ','_').replace('[','').replace(']','')


    WhichExperiment.HardParams.Model.MultiClass.num_classes = len(WhichExperiment.Nucleus.Index) + 1 if WhichExperiment.HardParams.Model.MultiClass.mode else 2


    directories = smallFuncs.search_ExperimentDirectory(WhichExperiment)
    if not Augment.Mode:  Augment.AugmentLength = 0
    preprocess.Cropping.Method = UserInfo['cropping_method']


    preprocess.Mode                = UserInfo['preprocessMode']
    preprocess.BiasCorrection.Mode = UserInfo['BiasCorrection']
    preprocess.Cropping.Mode       = UserInfo['Cropping']
    preprocess.Normalize.Mode      = UserInfo['Normalize']
    preprocess.TestOnly            = UserInfo['TestOnly']

    Augment.Mode                     = UserInfo['AugmentMode']  
    Augment.Linear.Rotation.Mode     = UserInfo['Augment_Rotation']
    Augment.Linear.Rotation.AngleMax = UserInfo['Augment_AngleMax']
    Augment.Linear.Shift.Mode        = UserInfo['Augment_Shift']
    Augment.Linear.Shift.ShiftMax    = UserInfo['Augment_ShiftMax']
    Augment.NonLinear.Mode           = UserInfo['Augment_NonLinearMode']
    # WhichExperiment.Dataset.CreatingTheExperiment = UserInfo['CreatingTheExperiment']

    params.WhichExperiment = WhichExperiment
    params.preprocess      = preprocess
    params.directories     = directories
    params.UserInfo        = UserInfo
    params.Augment         = Augment


    if preprocess.TestOnly:
        hist_params = pd.read_csv(directories.Train.Model + '/hist_params.csv').head()

        params.WhichExperiment.HardParams.Model.InputDimensions = [hist_params['InputDimensionsX'][0], hist_params['InputDimensionsY'][0],0]
        params.WhichExperiment.HardParams.Model.num_Layers = hist_params['num_Layers'][0]

    return params
