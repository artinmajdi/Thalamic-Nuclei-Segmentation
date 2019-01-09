import os, sys
__file__ = '/array/ssd/msmajdi/code/thalamus/keras/'  #! only if I'm using Hydrogen Atom
sys.path.append(os.path.dirname(__file__))
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import backend as K
import tensorflow as tf

from otherFuncs import smallFuncs, datasets, choosingModel
from Parameters import params
from preprocess import applyPreprocess

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


# params.preprocess.TestOnly = True
if not params.preprocess.TestOnly:
    #! Training the model
    model = choosingModel.architecture(params)
    model, hist = choosingModel.modelTrain(Data, params, model)
else:
    #! loading the model
    # modelFile = params.directories.Train.Model + '/model.h5'
    # model = model_from_json(open(modelFile).read())
    # model.load_weights(os.path.join(os.path.dirname(modelFile), 'model_weights.h5'))
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

