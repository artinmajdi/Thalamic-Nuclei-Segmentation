import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from otherFuncs import smallFuncs, datasets
from tqdm import tqdm
from modelFuncs import choosingModel
from keras.models import load_model
from copy import deepcopy


# TODO:   make a new functions for reading all dices for each test cases and put them in a table for each nuclei
# TODO:    write a new function taht could raed the history files and plot the dice, loss for trainign and validation

AllExperimentsList = {
    1: dict(),
    # 2: dict(nucleus_Index = [6] , GPU_Index = 6 , lossFunctionIx = 2),
    # 3: dict(nucleus_Index = [6] , GPU_Index = 7 , lossFunctionIx = 3),
    # 4: dict(nucleus_Index = [8] , GPU_Index = 5 , lossFunctionIx = 1),
    # 5: dict(nucleus_Index = [8] , GPU_Index = 6 , lossFunctionIx = 2),
    # 6: dict(nucleus_Index = [8] , GPU_Index = 7 , lossFunctionIx = 3),
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


#! we assume that number of layers , and other things that might effect the input data stays constant
AllParamsList = smallFuncs.readingTheParams(AllExperimentsList)

#! reading the dataset
ind, params = list(AllParamsList.items())[0]
print('Nuclei:',params.WhichExperiment.Nucleus.name , '  GPU:',params.WhichExperiment.HardParams.Machine.GPU_Index , \
'  Epochs:', params.WhichExperiment.HardParams.Model.epochs,'  Dataset:',params.WhichExperiment.Dataset.name , \
'  Experiment: {',params.WhichExperiment.Experiment.name ,',', params.WhichExperiment.SubExperiment.name,'}')

Data, params, Info = datasets.check_Dataset(params=params, flag=True, Info={})

for ind, params in list(AllParamsList.items()):

    K = gpuSetting(params)
    print('nuclei: ',params.WhichExperiment.Nucleus.name , 'gpu:',params.WhichExperiment.HardParams.Machine.GPU_Index)
    _, params, _ = datasets.check_Dataset(params=params, flag=False, Info=Info)

    pred = choosingModel.check_Run(params, Data)

    if 0: check_show(Data, pred)

K.clear_session()