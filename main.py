import os, sys
__file__ = '/array/ssd/msmajdi/code/thalamus/keras/'  #! only if I'm using Hydrogen Atom
sys.path.append(os.path.dirname(__file__))
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import backend as K
import tensorflow as tf
import mat4py

from otherFuncs import smallFuncs, datasets, choosingModel
from Parameters import UserInfo, paramFunc
from preprocess import applyPreprocess

params = paramFunc.__init__(UserInfo)
params.preprocess.Mode = False
params.preprocess.CreatingTheExperiment = False
mode = 'experiment'


# TODO: saving the param variable as a pickle file in the model output
params = smallFuncs.terminalEntries(params)
os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index


#! copying the dataset into the experiment folder
if params.preprocess.CreatingTheExperiment: datasets.movingFromDatasetToExperiments(params)


#! preprocessing the data
if params.preprocess.Mode:
    applyPreprocess.main(params, mode)
    params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)


#! configing the GPU
session = tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   )
K.set_session(session)


#! correcting the number of layers
params = smallFuncs.correctNumLayers(params)


#! Finding the final image sizes after padding & amount of padding
params = smallFuncs.imageSizesAfterPadding(params, mode)


# params.preprocess.TestOnly = True
#! loading the dataset
Data, params = datasets.loadDataset(params)


import pickle



# params.preprocess.TestOnly = True
if not params.preprocess.TestOnly:
    #! Training the model
    smallFuncs.Saving_UserInfo(params.directories.Train.Model, params, UserInfo)
    model = choosingModel.architecture(params)
    model, hist = choosingModel.modelTrain(Data, params, model)


    smallFuncs.saveReport(params.directories.Train.Model , 'hist_history' , hist.history , UserInfo.SaveReportMethod)
    smallFuncs.saveReport(params.directories.Train.Model , 'hist_model'   , hist.model   , UserInfo.SaveReportMethod)
    smallFuncs.saveReport(params.directories.Train.Model , 'hist_params'  , hist.params  , UserInfo.SaveReportMethod)

else:
    # TODO: I need to think more about this, why do i need to reload params even though i already have to load it in the beggining of the code
    #! loading the params
    UserInfo = smallFuncs.Loading_UserInfo(params.directories.Train.Model + '/UserInfo.mat', UserInfo.SaveReportMethod)
    params = paramFunc.__init__(UserInfo)
    params.WhichExperiment.HardParams.Model.InputDimensions = UserInfo.InputDimensions
    params.WhichExperiment.HardParams.Model.num_Layers      = UserInfo.num_Layers

    #! loading the model
    model = load_model(params.directories.Train.Model + '/model.h5')



#! Testing
pred, Dice, score = {}, {}, {}
for name in tqdm(Data.Test):
    ResultDir = params.directories.Test.Result
    padding = params.directories.Test.Input.Subjects[name].Padding
    Dice[name], pred[name], score[name] = choosingModel.applyTestImageOnModel(model, Data.Test[name], params, name, padding, ResultDir)



#! training predictions
if params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data:
    ResultDir = smallFuncs.mkDir(params.directories.Test.Result + '/TrainData_Output')
    for name in tqdm(Data.Train_ForTest):
        padding = params.directories.Train.Input.Subjects[name].Padding
        Dice[name], pred[name], score[name] = choosingModel.applyTestImageOnModel(model, Data.Train_ForTest[name], params, name, padding, ResultDir)



#! showing the outputs
for ind in [10]: # ,13,17]:
    name = list(Data.Test)[ind]   # Data.Train_ForTest
    # name = 'vimp2_2039_03182016'
    smallFuncs.imShow( Data.Test[name].Image[ind,:,:,0] ,  Data.Test[name].OrigMask[...,ind,0]  ,  pred[name][...,ind,0] )
    print(ind, name, Dice[name])

# K.clear_session()


