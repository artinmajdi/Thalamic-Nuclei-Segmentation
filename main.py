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
# TODO : look for a way to see epoch inside my loss function and use BCE initially and tyhen add Dice for higher epochs
UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
params = paramFunc.Run(UserInfoB)


def gpuSetting(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
    return K

def SingleNucleiRun(params):

    print(' ')
    print(' ')
    print('Nuclei:',params.WhichExperiment.Nucleus.name , '  GPU:',params.WhichExperiment.HardParams.Machine.GPU_Index , \
    '  Epochs:', params.WhichExperiment.HardParams.Model.epochs,'  Dataset:',params.WhichExperiment.Dataset.name , \
    '  Experiment: {',params.WhichExperiment.Experiment.name ,',', params.WhichExperiment.SubExperiment.name,'}')
    print(' ')
    print(' ')
        
    # Data, params, Info = datasets.check_Dataset_ForTraining(params=params, flag=True, Info={})
    Data, params = datasets.loadDataset(params)


    # mode = 'singleExperiment'
    # if 'singleExperiment' in mode:
    # K = runExperiment(params, Info, Data)
    # else:    
    #     for _, params in list(AllParamsList.items()): 
    #         K = runExperiment(params, Info, Data)

    K = gpuSetting(params)
    choosingModel.check_Run(params, Data)

    return K


#! copying the dataset into the experiment folder
datasets.movingFromDatasetToExperiments(params)


NucleiIndexes = UserInfoB['nucleus_Index']
for UserInfoB['nucleus_Index'] in NucleiIndexes: #  FullIndexes:
    # UserInfoB['nucleus_Index'] = [nucleiIx]
    params = paramFunc.Run(UserInfoB)
    K = SingleNucleiRun(params)


K.clear_session()