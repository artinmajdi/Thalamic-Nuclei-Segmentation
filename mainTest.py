import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# from keras_tqdm import TQDMCallback
import params
from choosingModel import architecture, modelTrain
import numpy as np
from datasets import loadDataset
from keras.models import load_model, Model
import matplotlib.pyplot as plt
from time import time
# from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

config = tf.ConfigProto( allow_soft_placement=True )  # intra_op_parallelism_threads=4,inter_op_parallelism_threads=20, 
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list='6'

session = tf.Session(config=config)
K.set_session(session)

Data = loadDataset( params.directories.Experiment.HardParams.Model )

pred = {}
for params.directories.Experiment.HardParams.Model.architectureType in ['U-Net', 'CNN_Segmetnation']:
    t = time()
    model, params = architecture(Data, params)
    model = modelTrain(Data, params, model)
    # model = load_model(params.directories.Train.Model + '/model.h5')
    pred[params.directories.Experiment.HardParams.Model.architectureType] = model.predict(Data.Test.Image)
    print(time() - t)



ind = 3
fig, axes = plt.subplots(1,4)
axes[2].imshow(np.squeeze(pred['U-Net'][ind,:,:,1]),cmap='gray')
axes[3].imshow(np.squeeze(pred['CNN_Segmetnation'][ind,:,:,0]),cmap='gray')#, title('pred CNN')
axes[1].imshow(np.squeeze(Data.Test.Label[ind,:,:,0]),cmap='gray')#, title('mask')
axes[0].imshow(np.squeeze(Data.Test.Image[ind,:,:,0]),cmap='gray')#, title('image')
plt.show()
print(time() - t)

pred
