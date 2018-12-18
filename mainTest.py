import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# from keras_tqdm import TQDMCallback
import params
from architectures import architecturesMain
import numpy as np
from datasets import loadDataset
from keras.models import load_model, Model
import matplotlib.pyplot as plt
# from tqdm import tqdm
# import tensorflow as tf
# from keras import backend as K

# config = tf.ConfigProto(allow_soft_placement=True ,\
#         device_count = {'GPU' : 1})
# session = tf.Session(config=config)
# K.set_session(session)


Data = loadDataset( params.directories.Experiment.HardParams.Model )
model = architecturesMain( params , Data )


# model = load_model(params.directories.Train.Model + '/model.h5')

pred = model.predict(Data.Test.Image)
pred.shape
ind = 3
fig, axes = plt.subplots(1,2)
axes[0].imshow(np.squeeze(pred[ind,:,:,0]),cmap='gray')
axes[1].imshow(np.squeeze(Data.Test.Label[ind,:,:,0]),cmap='gray')
plt.show()
