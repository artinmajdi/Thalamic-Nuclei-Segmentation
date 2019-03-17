import numpy as np
import nibabel as nib
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import otherFuncs.smallFuncs as smallFuncs
import modelFuncs.choosingModel as choosingModel
import otherFuncs.datasets as datasets
import pickle
import keras.models as kerasmodels
import keras

params = paramFunc.Run(UserInfo.__dict__)
Data, params = datasets.loadDataset(params)
model = choosingModel.architecture(params)
model.load_weights(params.directories.Train.Model + '/model_weights.h5')


def main(nl):
    print(model.layers[12].output)
    inputs = keras.layers.Input( tuple(params.WhichExperiment.HardParams.Model.InputDimensions[:2]) + (1,) )
    model2 = keras.models.Model(inputs=[model.layers[0].output], outputs=[model.layers[nl].output, model.layers[-1].output])

    subject = Data.Test[list(Data.Test)[0]]

    pred, final = model2.predict(subject.Image)

    a = nib.viewers.OrthoSlicer3D(pred,title='intermediate prediction')
    b = nib.viewers.OrthoSlicer3D(subject.Image,title='Image')
    c = nib.viewers.OrthoSlicer3D(final[...,0], title='final prediction')
    a.link_to(b)
    a.link_to(c)
    a.show()

main(12)
print('----')
