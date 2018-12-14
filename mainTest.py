# %%
import keras
from keras import optimizers
import params
import architectures
import numpy as np
from datasets import loadDataset

ModelParam = params.directories.Experiment.HardParams.Model

Train, Test = loadDataset(ModelParam.dataset)

if 'U-Net' in ModelParam.ArchitectureType:
    model = architectures.UNet( ModelParam )

elif 'MLP' in ModelParam.ArchitectureType:
    ModelParam.NumClasses = len(np.unique(Train.Label))
    model = architectures.MLP( ModelParam )

model.compile(optimizer=ModelParam.Optimizer,loss=ModelParam.loss,metrics=ModelParam.metrics)
model.fit(x=Train.Data,y=Train.Label,batch_size=ModelParam.batch_size,epochs=ModelParam.epochs)

print('---')
