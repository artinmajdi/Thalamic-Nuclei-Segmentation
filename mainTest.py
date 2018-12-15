import params
import architectures
import numpy as np
from datasets import loadDataset
from tqdm import tqdm
from keras_tqdm import TQDMCallback
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto( allow_soft_placement = True)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '4'

set_session(tf.Session(config=config))

ModelParam = params.directories.Experiment.HardParams.Model

Train, Test = loadDataset(ModelParam.dataset)


if 'U-Net' in ModelParam.architectureType:
    model = architectures.UNet(ModelParam)

elif 'MLP' in ModelParam.architectureType:
    ModelParam.numClasses = Train.Label.shape[1] # len(np.unique(Train.Label))
    model = architectures.CNN(ModelParam)

model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss, metrics=ModelParam.metrics)

# for mode in tqdm([model],desc="Training mode"):
model.fit(x=Train.Data, y=Train.Label, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs)

print('this is')
