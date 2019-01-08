import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Reshape, Flatten, BatchNormalization, Input, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras import losses
from otherFuncs import smallFuncs
from keras_tqdm import TQDMCallback # , TQDMNotebookCallback
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

# def Dice_Calculator(msk1,msk2):
#     return tf.reduce_sum(tf.multiply(msk1,msk2))*2/( tf.reduce_sum(msk1) + tf.reduce_sum(msk2) + tf.keras.backend.epsilon())



# TODO: check if the params includes the padding size for training and whether it saves it via pickle beside the model
# TODO save the params in the model folder
# ! main Function
def modelTrain(Data, params, model):
    ModelParam = params.WhichExperiment.HardParams.Model


    if params.WhichExperiment.Nucleus.Index[0] != 1 and params.WhichExperiment.HardParams.Model.InitializeFromThalamus and os.path.exists(params.directories.Train.Model_Thalamus + '/model_weights.h5'):
        model.load_weights(params.directories.Train.Model_Thalamus + '/model_weights.h5')
    elif params.WhichExperiment.HardParams.Model.InitializeFromOlderModel and os.path.exists(params.directories.Train.Model + '/model_weights.h5'):
        model.load_weights(params.directories.Train.Model + '/model_weights.h5')

    model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)

    # if the shuffle argument in model.fit is set to True (which is the default), the training data will be randomly shuffled at each epoch.
    if params.WhichExperiment.Dataset.Validation.fromKeras:
        hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs, shuffle=True, validation_split=params.WhichExperiment.Dataset.Validation.percentage, verbose=1, callbacks=[TQDMCallback()])
    else:
        hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Label), verbose=0, callbacks=[TQDMCallback()])

    smallFuncs.mkDir(params.directories.Train.Model)
    model.save(params.directories.Train.Model + '/model.h5', overwrite=True, include_optimizer=True )
    model.save_weights(params.directories.Train.Model + '/model_weights.h5', overwrite=True )
    if ModelParam.showHistory: print(hist.history)

    #! saving the params in the model folder
    # f = open(params.directories.Train.Model + '/params.pckl', 'wb')
    # pickle.dump(params, f)
    # f.close()

    return model, hist

def architecture(Data, params):
    # params.WhichExperiment.HardParams.Model.imageInfo = Data.Info
    ModelParam = params.WhichExperiment.HardParams.Model
    if 'U-Net' in ModelParam.architectureType:
        model = UNet(ModelParam)

    elif 'CNN_Segmetnation' in ModelParam.architectureType:
        model = CNN_Segmetnation(ModelParam)

    elif 'CNN_Classifier' in ModelParam.architectureType:
        ModelParam.numClasses = Data.Train.Mask.shape[1] # len(np.unique(Train.Label))
        model = CNN_Segmetnation(ModelParam)

    model.summary()

    # ModelParam = params.WhichExperiment.HardParams.Model
    # model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
    return model

def Unet_sublayer_Contracting(inputs, nL, Modelparam):
    if Modelparam.batchNormalization:  inputs = BatchNormalization()(inputs)
    conv = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(inputs)
    conv = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)
    pool = MaxPooling2D( pool_size=Modelparam.MaxPooling.pool_size)(conv)
    if Modelparam.Dropout.Mode: pool = Dropout(Modelparam.Dropout.Value)(pool)
    return pool, conv

def Unet_sublayer_Expanding(inputs, nL, Modelparam, contractingInfo):
    if Modelparam.batchNormalization:  inputs = BatchNormalization()(inputs)
    UP = Conv2DTranspose(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.convTranspose, strides=(2,2), padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(inputs)
    UP = concatenate([UP,contractingInfo[nL+1]],axis=3)
    conv = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(UP)
    conv = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)
    if Modelparam.Dropout.Mode: conv = Dropout(Modelparam.Dropout.Value)(conv)
    return conv

# ! U-Net Architecture
def UNet(Modelparam):
    inputs = Input( (Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1) )
    WeightBiases = inputs

    # ! contracting layer
    ConvOutputs = {}
    for nL in range(Modelparam.num_Layers -1):
        WeightBiases, ConvOutputs[nL+1] = Unet_sublayer_Contracting(WeightBiases, nL, Modelparam)

    # ! middle layer
    nL = Modelparam.num_Layers - 1
    WeightBiases = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(WeightBiases)
    WeightBiases = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(WeightBiases)
    if Modelparam.Dropout.Mode: WeightBiases = Dropout(Modelparam.Dropout.Value)(WeightBiases)

    # ! expanding layer
    for nL in reversed(range(Modelparam.num_Layers -1)):
        WeightBiases = Unet_sublayer_Expanding(WeightBiases, nL, Modelparam, ConvOutputs)

    # ! final outputing the data
    final = Conv2D(2, kernel_size=Modelparam.ConvLayer.Kernel_size.output, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.output)(WeightBiases)
    # final = Conv2D(1, kernel_size=Modelparam.ConvLayer.Kernel_size.output, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.output)(WeightBiases)
    model = Model(inputs=[inputs], outputs=[final])

    return model

# ! CNN Architecture
def CNN_Segmetnation(Modelparam):
    inputs = Input( (Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1) )
    conv = inputs

    for nL in range(Modelparam.num_Layers -1):
        conv = Conv2D(filters=64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)
        conv = Dropout(Modelparam.Dropout.Value)(conv)

    final  = Conv2D(filters=2, kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)

    model = Model(inputs=[inputs], outputs=[final])

    return model

# ! CNN Architecture
def CNN_Classifier(Modelparam):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape=(Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1)))
    model.add(MaxPooling2D(pool_size=Modelparam.MaxPooling.pool_size))
    model.add(Dropout(Modelparam.Dropout.Value))

    model.add(Conv2D(filters=8, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape=(Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1)))
    model.add(MaxPooling2D(pool_size=Modelparam.MaxPooling.pool_size))
    model.add(Dropout(Modelparam.Dropout.Value))

    model.add(Flatten())
    model.add(Dense(128 , activation=Modelparam.Activitation.layers))
    model.add(Dropout(Modelparam.Dropout.Value))
    model.add(Dense(Modelparam.numClasses , activation=Modelparam.Activitation.output))

    return model

#! training on multiple architecture
def trainingMultipleMethod(params, Data):
    Models, Hist = {}, {}
    for params.WhichExperiment.HardParams.Model.architectureType in tqdm(['U-Net' , 'CNN_Segmetnation']):
        architectureType = params.WhichExperiment.HardParams.Model.architectureType
        model = architecture(Data, params)
        Models[architectureType], Hist[architectureType] = modelTrain(Data, params, model)

    return Models, Hist

#! applying the trained model on test data
def applyTestImageOnModel(model, Data, params, name, padding, ResultDir):
    pred = model.predict(Data.Image)
    score = model.evaluate(Data.Image, Data.Mask)
    pred = np.transpose(pred,[1,2,0,3])[...,0]

    Thresh = max( filters.threshold_otsu(pred) ,0.2)  if len(np.unique(pred)) != 1 else 0
    # Thresh = 0.2

    pred = smallFuncs.unPadding(pred, padding)  > Thresh
    smallFuncs.saveImage(pred, Data.Affine, Data.Header, ResultDir + '/' + params.WhichExperiment.Nucleus.name + '_' + name + '_pred.nii.gz')
    Dice = smallFuncs.Dice_Calculator(pred , Data.OrigMask)
    np.savetxt(ResultDir + '/' +  params.WhichExperiment.Nucleus.name + '_' + name + '_Dice.txt',[Dice])

    return Dice, pred, score
