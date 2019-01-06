import os, sys
__file__ = '/array/ssd/msmajdi/code/thalamus/keras/'  #! only if I'm using Hydrogen Atom
sys.path.append(os.path.dirname(__file__))
import numpy as np
from keras.models import load_model, Model, model_from_json
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from keras import backend as K
import tensorflow as tf
import pickle
import nibabel as nib

from otherFuncs import params, smallFuncs, datasets, choosingModel
from preprocess.preprocessA import main_preprocess
params.preprocess.Mode = False
params.preprocess.CreatingTheExperiment = False
mode = 'experiment'

def Dice_Calculator(msk1,msk2):
    return tf.reduce_sum(tf.multiply(msk1,msk2))*2/( tf.reduce_sum(msk1) + tf.reduce_sum(msk2) + tf.keras.backend.epsilon())


# TODO: check the new conda environement with skimage to make sure it works
# TODO: saving the param variable as a pickle file in the model output
params = smallFuncs.terminalEntries(params)
print(params.WhichExperiment.HardParams.Model.epochs)
os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index


#! copying the dataset into the experiment folder
if params.preprocess.CreatingTheExperiment: datasets.movingFromDatasetToExperiments(params)


#! preprocessing the data
if params.preprocess.Mode:
    main_preprocess(params, mode)
    params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)


#! configing the GPU
session = tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   )
K.set_session(session)


#! correcting the number of layers
params = smallFuncs.correctNumLayers(params)


#! Finding the final image sizes after padding & amount of padding
params = smallFuncs.imageSizesAfterPadding(params, mode)

# params.preprocess.TestOnly = True
# params.preprocess.TestOnly = False
#! loading the dataset
Data, params = datasets.loadDataset(params)


# params.preprocess.TestOnly = True
if not params.preprocess.TestOnly:
    #! Training the model
    model = choosingModel.architecture(Data, params)
    model, hist = choosingModel.modelTrain(Data, params, model)
else:
    #! loading the model
    # modelFile = params.directories.Train.Model + '/model.h5'
    # model = model_from_json(open(modelFile).read())
    # model.load_weights(os.path.join(os.path.dirname(modelFile), 'model_weights.h5'))
    model = load_model(params.directories.Train.Model + '/model.h5')


#! Testing
pred, Dice, score = {}, {}, {}
for ind, name in tqdm(enumerate(Data.Test)):
    ResultDir = params.directories.Test.Result
    padding = params.directories.Test.Input.Subjects[name].Padding
    Dice[name], pred[name], score[name] = choosingModel.applyTestImageOnModel(model, Data.Test[name], params, name, padding, ResultDir)

#! training predictions
for ind, name in tqdm(enumerate(Data.Train_ForTest)):
    ResultDir = params.directories.Test.Result
    padding = params.directories.Train.Input.Subjects[name].Padding
    Dice[name], pred[name], score[name] = choosingModel.applyTestImageOnModel(model, Data.Train_ForTest[name], params, name, padding, ResultDir)

#! showing the outputs
for ind in [10,13,17]:
    name = list(Data.Test)[ind]
    # name = 'vimp2_2039_03182016'
    smallFuncs.imShow( Data.Test[name].Image[ind,:,:,0] ,  Data.Test[name].OrigMask[...,ind]  ,  pred[name][...,ind] )
    print(ind, name, Dice[name])

#! showing the outputs
for ind in [10,13,17]:
    name = list(Data.Train_ForTest)[ind]
    # name = 'vimp2_2039_03182016'
    smallFuncs.imShow( Data.Train_ForTest[name].Image[ind,:,:,0] ,  Data.Train_ForTest[name].OrigMask[...,ind]  ,  pred[name][...,ind] )
    print(ind, name, Dice[name])

# K.clear_session()


