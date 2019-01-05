import os, sys
__file__ = '/array/ssd/msmajdi/code/thalamus/keras/'  #! only if I'm using Hydrogen Atom
sys.path.append(os.path.dirname(__file__))
import numpy as np
from keras.models import load_model, Model
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

# TODO: saving the param variable as a pickle file in the model output
params = smallFuncs.terminalEntries(params)
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


#! loading the dataset
Data, params = datasets.loadDataset(params)


if 0:
    #! Training
    model = choosingModel.architecture(Data, params)
    model, hist = choosingModel.modelTrain(Data, params, model)
else:
    #! Testing
    model = load_model(params.directories.Train.Model + '/model.h5')

pred, Dice = {}, {}
for ind, name in tqdm(enumerate(Data.Test)):
    Dice[name], pred[name] = choosingModel.applyTestImageOnModel(model, Data.Test[name], params, name)
    print(ind, name, Dice[name])
    #! showing the outputs
    ind = 2
    if 0: smallFuncs.imShow( Data.Test[name].Image[ind,:,:,0] ,  Data.Test[name].OrigMask[...,ind]  ,  pred[name][...,ind] )

K.clear_session()
