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
    directories     = ''
    UserInfo        = ''
    
def Run(UserInfoB):


    params = deepcopy(paramsA)
    UserInfo = deepcopy(UserInfoB)

    WhichExperiment = deepcopy(params.WhichExperiment)
    preprocess      = deepcopy(params.preprocess)

    WhichExperiment.address = smallFuncs.mkDir(UserInfo['Experiments_Address'])

    WhichExperiment.HardParams.Template.Image = UserInfo['Tempalte_Image']
    WhichExperiment.HardParams.Template.Mask  = UserInfo['Tempalte_Mask']
    WhichExperiment.HardParams.Model.MultiClass.mode = UserInfo['MultiClass_mode']
    WhichExperiment.HardParams.Model.loss, _      = LossFunction.LossInfo(UserInfo['lossFunctionIx'])
    WhichExperiment.HardParams.Model.metrics, _   = Metrics.MetricInfo(UserInfo['MetricIx'])
    WhichExperiment.HardParams.Model.optimizer, _ = Optimizers.OptimizerInfo(UserInfo['OptimizerIx'])
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


    WhichExperiment.SubExperiment.index = UserInfo['SubExperiment_Index']
    WhichExperiment.Experiment.index = UserInfo['Experiments_Index']
    WhichExperiment.Experiment.tag = UserInfo['Experiments_Tag']
    WhichExperiment.Experiment.name = 'exp' + str(UserInfo['Experiments_Index']) + '_' + WhichExperiment.Experiment.tag if WhichExperiment.Experiment.tag else 'exp' + str(WhichExperiment.Experiment.index)
    WhichExperiment.Experiment.address = smallFuncs.mkDir(WhichExperiment.address + '/' + WhichExperiment.Experiment.name)
    _, B = LossFunction.LossInfo(UserInfo['lossFunctionIx']) 
    WhichExperiment.SubExperiment.tag = UserInfo['SubExperiment_Tag'] + B
    # WhichExperiment.SubExperiment.name = 'subExp' + str(WhichExperiment.SubExperiment.index) + '_' + WhichExperiment.SubExperiment.tag + WhichExperiment.Nucleus.name if WhichExperiment.SubExperiment.tag else 'subExp' + str(WhichExperiment.SubExperiment.index) + '_' + WhichExperiment.Nucleus.name
    WhichExperiment.SubExperiment.name = 'subExp' + str(WhichExperiment.SubExperiment.index) + '_' + WhichExperiment.SubExperiment.tag if WhichExperiment.SubExperiment.tag else 'subExp' + str(WhichExperiment.SubExperiment.index) 

    # WhichExperiment.SubExperiment.name_thalamus = 'subExp' + str(WhichExperiment.SubExperiment.index) + '_' + WhichExperiment.SubExperiment.tag if WhichExperiment.SubExperiment.tag else 'subExp' + str(WhichExperiment.SubExperiment.index)


    # TODO I need to fix this to count for multiple nuclei
    WhichExperiment.Nucleus.Index = UserInfo['nucleus_Index'] # if WhichExperiment.HardParams.Model.MultiClass.mode else UserInfo['nucleus_Index']
    WhichExperiment.Nucleus.name_Thalamus, WhichExperiment.Nucleus.FullIndexes = smallFuncs.NucleiSelection( 1 , WhichExperiment.Nucleus.Organ)
    if len(WhichExperiment.Nucleus.Index) == 1:
        WhichExperiment.Nucleus.name , _ = smallFuncs.NucleiSelection( WhichExperiment.Nucleus.Index[0] , WhichExperiment.Nucleus.Organ)
    else:
        WhichExperiment.Nucleus.name = ('MultiClass_' + str(WhichExperiment.Nucleus.Index)).replace(', ','_').replace('[','').replace(']','')


    WhichExperiment.HardParams.Model.MultiClass.num_classes = len(WhichExperiment.Nucleus.Index) + 1 if WhichExperiment.HardParams.Model.MultiClass.mode else 2


    directories = smallFuncs.funcExpDirectories(WhichExperiment)
    preprocess.Augment = smallFuncs.augmentLengthChecker(preprocess.Augment)
    preprocess.Cropping.Method = smallFuncs.whichCropMode(WhichExperiment.Nucleus.name, UserInfo['cropping_method'])  # it changes the mode to 1 if we're analyzing the Thalamus

    
    preprocess.Mode                = UserInfo['preprocessMode']
    preprocess.BiasCorrection.Mode = UserInfo['BiasCorrection']
    preprocess.Cropping.Mode       = UserInfo['Cropping']
    preprocess.Normalize.Mode      = UserInfo['Normalize']
    preprocess.Augment.Mode        = UserInfo['AugmentMode']
    preprocess.TestOnly            = UserInfo['TestOnly']

    preprocess.Augment.Rotation     = UserInfo['Augment_Rotation']
    preprocess.Augment.Shift        = UserInfo['Augment_Shift']
    preprocess.Augment.NonRigidWarp = UserInfo['Augment_NonRigidWarp']
    preprocess.CreatingTheExperiment = UserInfo['CreatingTheExperiment']
    
    params.WhichExperiment = WhichExperiment
    params.preprocess      = preprocess
    params.directories     = directories
    params.UserInfo        = UserInfo

    if preprocess.TestOnly:
        hist_params = pd.read_csv(directories.Train.Model + '/hist_params.csv').head()

        params.WhichExperiment.HardParams.Model.InputDimensions = [hist_params['InputDimensionsX'][0], hist_params['InputDimensionsY'][0],0]
        params.WhichExperiment.HardParams.Model.num_Layers = hist_params['num_Layers'][0]

    return params

