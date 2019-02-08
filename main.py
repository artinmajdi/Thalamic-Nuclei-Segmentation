import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from otherFuncs import smallFuncs, datasets
from tqdm import tqdm
from modelFuncs import choosingModel
from keras.models import load_model
from copy import deepcopy
from Parameters import UserInfo, paramFunc

# TODO:  add a fixed seed number for random numbers 
# TODO:  write the name of test and train subjects in model and results and dataset  to have it for the future
# TODO:   make a new functions for reading all dices for each test cases and put them in a table for each nuclei
# TODO:    write a new function taht could raed the history files and plot the dice, loss for trainign and validation
# TODO : look for a way to see epoch inside my loss function and use BCE initially and tyhen add Dice for higher epochs
# TODO: use linear rotation augmentation

UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)

# AllExperimentsList = {
#     1: dict(),
#     # 2: dict(nucleus_Index = [6] , GPU_Index = 6 , lossFunctionIx = 2),
# }

def gpuSetting(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
    return K

# def runExperiment(params, Info, Data):

#     # params.WhichExperiment.Dataset.CreatingTheExperiment = False

#     K = gpuSetting(params)
#     pred = choosingModel.check_Run(params, Data)

#     return K

def SingleNucleiRun(params):

    print('Nuclei:',params.WhichExperiment.Nucleus.name , '  GPU:',params.WhichExperiment.HardParams.Machine.GPU_Index , \
    '  Epochs:', params.WhichExperiment.HardParams.Model.epochs,'  Dataset:',params.WhichExperiment.Dataset.name , \
    '  Experiment: {',params.WhichExperiment.Experiment.name ,',', params.WhichExperiment.SubExperiment.name,'}')
    
        
    # Data, params, Info = datasets.check_Dataset_ForTraining(params=params, flag=True, Info={})
    params.WhichExperiment.HardParams.Model.num_Layers = datasets.correctNumLayers(params)
    params = datasets.imageSizesAfterPadding(params, 'experiment')
    Data, params = datasets.loadDataset(params)


    # mode = 'singleExperiment'
    # if 'singleExperiment' in mode:
    # K = runExperiment(params, Info, Data)
    # else:    
    #     for _, params in list(AllParamsList.items()): 
    #         K = runExperiment(params, Info, Data)

    K = gpuSetting(params)
    pred = choosingModel.check_Run(params, Data)

    return K




if 1:

    #! we assume that number of layers , and other things that might effect the input data stays constant
    # AllParamsList = smallFuncs.readingTheParams(AllExperimentsList)
    # ind, params = list(AllParamsList.items())[0]
    params = paramFunc.Run(UserInfoB)
    K = SingleNucleiRun(params)

else:

    #! this is temporary to run on all nuclei
    # from Parameters import UserInfo, paramFunc
    # UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
    _, FullIndexes = smallFuncs.NucleiSelection(ind = 1,organ = 'THALAMUS')

    NucleiIndexes = UserInfoB['nucleus_Index']
    for nucleiIx in NucleiIndexes: #  FullIndexes:
        UserInfoB['nucleus_Index'] = [nucleiIx]
        params = paramFunc.Run(UserInfoB)
        K = SingleNucleiRun(params)


K.clear_session()