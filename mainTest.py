# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# from keras_tqdm import TQDMCallback
import params
from architectures import architecturesMain
import numpy as np
from datasets import loadDataset
# from tqdm import tqdm
# import tensorflow as tf
# from keras import backend as K

# config = tf.ConfigProto(allow_soft_placement=True ,\
#         device_count = {'GPU' : 1})
# session = tf.Session(config=config)
# K.set_session(session)


ModelParam = params.directories.Experiment.HardParams.Model
Train, Test, Info = loadDataset(ModelParam.dataset)
ModelParam.imageInfo = Info
# Train, Test = 0, 2
model = architecturesMain(ModelParam, Train, Test)
# model = get_unet()


print('this is')
