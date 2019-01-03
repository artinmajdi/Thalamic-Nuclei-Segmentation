import os, sys
__file__ = '/array/ssd/msmajdi/code/Thalamus_Keras/mainTest.py'  #! only if I'm using Hydrogen Atom
sys.path.append(os.path.dirname(__file__))
import numpy as np
from keras.models import load_model, Model
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from keras import backend as K
import tensorflow as tf
import pickle

from otherFuncs import params, smallFuncs, datasets, choosingModel
from preprocess.preprocessA import main_preprocess
params.preprocess.Mode = False
params.preprocess.CreatingTheExperiment = False


# TODO: saving the param variable as a pickle file in the model output
params = smallFuncs.terminalEntries(params)
os.environ["CUDA_VISIBLE_DEVICES"] = params.directories.WhichExperiment.HardParams.Machine.GPU_Index


#! copying the dataset into the experiment folder
if params.preprocess.CreatingTheExperiment: datasets.movingFromDatasetToExperiments(params)

#! preprocessing the data
if params.preprocess.Mode: main_preprocess(params, 'experiment')
params.directories = smallFuncs.funcExpDirectories(params.directories.WhichExperiment)


#! configing the GPU
session = tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   )
K.set_session(session)


#! correcting the number of layers
params = smallFuncs.correctNumLayers(params)


#! Finding the final image sizes after padding & amount of padding
params = smallFuncs.imageSizesAfterPadding(params)

#! loading the dataset
Data, params = datasets.loadDataset(params)



#! Actual architecture
pred = {}
for params.directories.WhichExperiment.HardParams.Model.architectureType in tqdm(['U-Net']):# , 'CNN_Segmetnation']):
    t = time()
    model, params = choosingModel.architecture(Data, params)
    model, hist   = choosingModel.modelTrain(Data, params, model)
    # model = load_model(params.directories.Train.Model + '/model.h5')
    pred[params.directories.WhichExperiment.HardParams.Model.architectureType] = model.predict(Data.Test.Image)
    print(time() - t)




# TODO check the matlab imshow3D see if i can use it in python
#! showing the outputs
ind = 2
L = len(list(pred))
if L == 1:
    smallFuncs.imShow( Data.Test.Image[ind,:,:,0] ,  Data.Test.Label[ind,:,:,0]  ,  pred[list(pred)[0]][ind,:,:,1] )
elif L == 2:
    smallFuncs.imShow( Data.Test.Image[ind,:,:,0] ,  Data.Test.Label[ind,:,:,0]  ,  pred[list(pred)[0]][ind,:,:,1]  ,  pred[list(pred)[1]][ind,:,:,1] )


K.clear_session()
