from choosingModel import architecture, modelTrain
import numpy as np
from datasets import loadDataset
from keras.models import load_model, Model
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from keras import backend as K
import params
import os
import tensorflow as tf
from otherFuncs.smallFuncs import terminalEntries , inputNamesCheck
params = terminalEntries(params)
params = inputNamesCheck(params)
os.environ["CUDA_VISIBLE_DEVICES"] = params.directories.Experiment.HardParams.Machine.GPU_Index


Subjects = params.directories.Train.Input.Subjects
for sj in Subjects:
    subject = Subjects[sj]
    im = nib.load(subject.Address + '/' subject.ImageProcessed + '.nii.gz')
    mask = nib.load(subject.Label.Address + '/' + subject.Label.LabelProcessed + '.nii.gz')
    break

subject = Subjects[sj]

subject.Label.Address + '/' + subject.Label.LabelProcessed + '.nii.gz'

session = tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   )
K.set_session(session)
Data = loadDataset( params )

pred = {}
for params.directories.Experiment.HardParams.Model.architectureType in tqdm(['U-Net']): # , 'CNN_Segmetnation']):
    t = time()
    model, params = architecture(Data, params)
    model, hist   = modelTrain(Data, params, model)
    # model = load_model(params.directories.Train.Model + '/model.h5')
    pred[params.directories.Experiment.HardParams.Model.architectureType] = model.predict(Data.Test.Image)
    print(time() - t)


ind = 3
Figs = [  np.squeeze(Data.Test.Image[ind,:,:,0])  ,  np.squeeze(Data.Test.Label[ind,:,:,0])  ,   np.squeeze(pred['U-Net'][ind,:,:,1])  ]

fig, axes = plt.subplots(1,len(Figs))
for sh in range(len(Figs)):
    axes[sh].imshow(Figs[sh],cmap='gray')

plt.show()
print(time() - t)

K.clear_session()
