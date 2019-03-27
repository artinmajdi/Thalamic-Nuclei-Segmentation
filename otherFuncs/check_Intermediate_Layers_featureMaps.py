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
import matplotlib.pyplot as plt

def gpuSetting(params):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
    return K
    
UserInfoB = UserInfo.__dict__
UserInfoB['simulation'].TestOnly = True
params = paramFunc.Run(UserInfoB)

K = gpuSetting(params)

Data, params = datasets.loadDataset(params)
params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data = False


model = choosingModel.architecture(params)
model.load_weights('/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/models/sE10_CascadewRot7_6cnts_sd1_Dt0.3_LR0.001_MpByTH_WoET/1-THALAMUS/model_weights.h5')
# model.load_weights(params.directories.Train.Model + '/model_weights.h5')


def main(nl):
    print(model.layers[12].output)
    dim = params.WhichExperiment.HardParams.Model.Method.InputImage2Dvs3D
    inputs = keras.layers.Input( tuple(params.WhichExperiment.HardParams.Model.InputDimensions[:dim]) + (1,) )
    model2 = keras.models.Model(inputs=[model.layers[0].output], outputs=[model.layers[nl].output, model.layers[-1].output])

    subject = Data.Test[list(Data.Test)[0]]

    pred, final = model2.predict(subject.Image)
    aa = np.ceil(np.sqrt(pred.shape[3]))
    imm = np.squeeze(pred[:,:,10,:])
    newshape = tuple([int(np.ceil(np.sqrt(imm.shape[2]))*np.array(imm.shape[i])) for i in range(2)])
    imm2 = imm.transpose(2,0,1).reshape(-1, imm.shape[1])
    plt.imshow(imm2,[])
    plt.show()
    print('---')
    # a = nib.viewers.OrthoSlicer3D(pred,title='intermediate prediction')
    # b = nib.viewers.OrthoSlicer3D(subject.Image,title='Image')
    # c = nib.viewers.OrthoSlicer3D(final[...,0], title='final prediction')
    # a.link_to(b)
    # a.link_to(c)
    # a.show()

main(24)
print('----')


K.clear_session()
