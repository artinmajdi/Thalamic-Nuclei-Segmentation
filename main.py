import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from otherFuncs import smallFuncs, datasets
from tqdm import tqdm
from modelFuncs import choosingModel
from keras.models import load_model
from copy import deepcopy

# TODO:  add a fixed seed number for random numbers 
# TODO:  write the name of test and train subjects in model and results and dataset  to have it for the future
# TODO:   make a new functions for reading all dices for each test cases and put them in a table for each nuclei
# TODO:    write a new function taht could raed the history files and plot the dice, loss for trainign and validation
# TODO : look for a way to see epoch inside my loss function and use BCE initially and tyhen add Dice for higher epochs
# TODO: use linear rotation augmentation

AllExperimentsList = {
    1: dict(),
    # 2: dict(nucleus_Index = [6] , GPU_Index = 6 , lossFunctionIx = 2),
}

def check_show(Data, pred):
    #! showing the outputs
    for ind in [10]: # ,13,17]:
        name = list(Data.Test)[ind]   # Data.Train_ForTest
        # name = 'vimp2_2039_03182016'
        smallFuncs.imShow( Data.Test[name].Image[ind,:,:,0] ,  Data.Test[name].OrigMask[...,ind,0]  ,  pred[name][...,ind,0] )

def gpuSetting(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
    return K

def runExperiment(params, Info, Data):

    # params.WhichExperiment.Dataset.CreatingTheExperiment = False


    print('Nuclei:',params.WhichExperiment.Nucleus.name , '  GPU:',params.WhichExperiment.HardParams.Machine.GPU_Index , \
    '  Epochs:', params.WhichExperiment.HardParams.Model.epochs,'  Dataset:',params.WhichExperiment.Dataset.name , \
    '  Experiment: {',params.WhichExperiment.Experiment.name ,',', params.WhichExperiment.SubExperiment.name,'}')


    K = gpuSetting(params)
    print('nuclei: ',params.WhichExperiment.Nucleus.name , 'gpu:',params.WhichExperiment.HardParams.Machine.GPU_Index)
    _, params, _ = datasets.check_Dataset_ForTraining(params=params, flag=False, Info=Info)

    pred = choosingModel.check_Run(params, Data)

    if 0: check_show(Data, pred)

    return K

def SingleNucleiRun(params):
    Data, params, Info = datasets.check_Dataset_ForTraining(params=params, flag=True, Info={})
    mode = 'singleExperiment'

    if 'singleExperiment' in mode:
        K = runExperiment(params, Info, Data)
    else:    
        for _, params in list(AllParamsList.items()): 
            K = runExperiment(params, Info, Data)

    return K


if 0:

    #! we assume that number of layers , and other things that might effect the input data stays constant
    AllParamsList = smallFuncs.readingTheParams(AllExperimentsList)
    #! reading the dataset
    ind, params = list(AllParamsList.items())[0]
    K = SingleNucleiRun(params)

else:

    #! this is temporary to run on all nuclei
    from Parameters import UserInfo, paramFunc
    UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
    _, FullIndexes = smallFuncs.NucleiSelection(ind = 1,organ = 'THALAMUS')

    for nucleiIx in FullIndexes:
        UserInfoB['nucleus_Index'] = [nucleiIx]
        params = paramFunc.Run(UserInfoB)
        K = SingleNucleiRun(params)


K.clear_session()