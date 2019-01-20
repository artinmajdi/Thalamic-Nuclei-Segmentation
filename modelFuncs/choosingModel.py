import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras import models as kerasmodels
from keras import layers
# from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback # , TQDMNotebookCallback
import numpy as np
from skimage.filters import threshold_otsu
from Parameters import paramFunc
from otherFuncs import smallFuncs
from tqdm import tqdm
from time import time

def check_Run(params, Data):

    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index

    # params.preprocess.TestOnly = True
    if not params.preprocess.TestOnly:
        #! Training the model
        a = time()
        smallFuncs.Saving_UserInfo(params.directories.Train.Model, params, params.UserInfo)
        model = architecture(params)
        model, hist = modelTrain(Data, params, model)
        hist.params['trainingTime'] = time() - a

        smallFuncs.saveReport(params.directories.Train.Model , 'hist_history' , hist.history , params.UserInfo['SaveReportMethod'])
        smallFuncs.saveReport(params.directories.Train.Model , 'hist_model'   , hist.model   , params.UserInfo['SaveReportMethod'])
        smallFuncs.saveReport(params.directories.Train.Model , 'hist_params'  , hist.params  , 'csv')
        

    else:
        # TODO: I need to think more about this, why do i need to reload params even though i already have to load it in the beggining of the code
        #! loading the params
        params.UserInfo = smallFuncs.Loading_UserInfo(params.directories.Train.Model + '/UserInfo.mat', params.UserInfo['SaveReportMethod'])
        params = paramFunc.Run(params.UserInfo)
        params.WhichExperiment.HardParams.Model.InputDimensions = params.UserInfo['InputDimensions']
        params.WhichExperiment.HardParams.Model.num_Layers      = params.UserInfo['num_Layers']

        #! loading the model
        model = kerasmodels.load_model(params.directories.Train.Model + '/model.h5')



    #! Testing
    pred, Dice, score = {}, {}, {}
    for name in tqdm(Data.Test):
        ResultDir = params.directories.Test.Result
        padding = params.directories.Test.Input.Subjects[name].Padding
        Dice[name], pred[name], score[name] = applyTestImageOnModel(model, Data.Test[name], params, name, padding, ResultDir)



    #! training predictions
    if params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data:
        ResultDir = smallFuncs.mkDir(params.directories.Test.Result + '/TrainData_Output')
        for name in tqdm(Data.Train_ForTest):
            padding = params.directories.Train.Input.Subjects[name].Padding
            Dice[name], pred[name], score[name] = applyTestImageOnModel(model, Data.Train_ForTest[name], params, name, padding, ResultDir)

    return pred




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

    return model, hist

def architecture(params):
    # params.WhichExperiment.HardParams.Model.imageInfo = Data.Info
    ModelParam = params.WhichExperiment.HardParams.Model
    if 'U-Net' in ModelParam.architectureType:
        model = UNet(ModelParam)

    elif 'CNN_Segmetnation' in ModelParam.architectureType:
        model = CNN_Segmetnation(ModelParam)

    elif 'CNN_Classifier' in ModelParam.architectureType:
        model = CNN_Segmetnation(ModelParam)

    model.summary()

    # ModelParam = params.WhichExperiment.HardParams.Model
    # model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
    return model

def Unet_sublayer_Contracting(inputs, nL, Modelparam):
    if Modelparam.batchNormalization:  inputs = layers.BatchNormalization()(inputs)
    conv = layers.Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(inputs)
    conv = layers.Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)
    pool = layers.MaxPooling2D( pool_size=Modelparam.MaxPooling.pool_size)(conv)
    if Modelparam.Dropout.Mode: pool = layers.Dropout(Modelparam.Dropout.Value)(pool)
    return pool, conv

def Unet_sublayer_Expanding(inputs, nL, Modelparam, contractingInfo):
    if Modelparam.batchNormalization:  inputs = layers.BatchNormalization()(inputs)
    UP = layers.Conv2DTranspose(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.convTranspose, strides=(2,2), padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(inputs)
    UP = layers.merge.concatenate([UP,contractingInfo[nL+1]],axis=3)
    conv = layers.Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(UP)
    conv = layers.Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)
    if Modelparam.Dropout.Mode: conv = layers.Dropout(Modelparam.Dropout.Value)(conv)
    return conv

# ! U-Net Architecture
def UNet(Modelparam):
    inputs = layers.Input( (Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1) )
    WeightBiases = inputs

    # ! contracting layer
    ConvOutputs = {}
    for nL in range(Modelparam.num_Layers -1):
        WeightBiases, ConvOutputs[nL+1] = Unet_sublayer_Contracting(WeightBiases, nL, Modelparam)

    # ! middle layer
    nL = Modelparam.num_Layers - 1
    WeightBiases = layers.Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(WeightBiases)
    WeightBiases = layers.Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(WeightBiases)
    if Modelparam.Dropout.Mode: WeightBiases = layers.Dropout(Modelparam.Dropout.Value)(WeightBiases)

    # ! expanding layer
    for nL in reversed(range(Modelparam.num_Layers -1)):
        WeightBiases = Unet_sublayer_Expanding(WeightBiases, nL, Modelparam, ConvOutputs)

    # ! final outputing the data
    final = layers.Conv2D(Modelparam.MultiClass.num_classes, kernel_size=Modelparam.ConvLayer.Kernel_size.output, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.output)(WeightBiases)
    # final = layers.Conv2D(1, kernel_size=Modelparam.ConvLayer.Kernel_size.output, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.output)(WeightBiases)
    model = kerasmodels.Model(inputs=[inputs], outputs=[final])

    return model

# ! CNN Architecture
def CNN_Segmetnation(Modelparam):
    inputs = layers.Input( (Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1) )
    conv = inputs

    for nL in range(Modelparam.num_Layers -1):
        conv = layers.Conv2D(filters=64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)
        conv = layers.Dropout(Modelparam.Dropout.Value)(conv)

    final  = layers.Conv2D(filters=Modelparam.MultiClass.num_classes, kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)

    model = kerasmodels.Model(inputs=[inputs], outputs=[final])

    return model

# ! CNN Architecture
def CNN_Classifier(Modelparam):
    model = kerasmodels.Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape=(Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1)))
    model.add(layers.MaxPooling2D(pool_size=Modelparam.MaxPooling.pool_size))
    model.add(layers.Dropout(Modelparam.Dropout.Value))

    model.add(layers.Conv2D(filters=8, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape=(Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1)))
    model.add(layers.MaxPooling2D(pool_size=Modelparam.MaxPooling.pool_size))
    model.add(layers.Dropout(Modelparam.Dropout.Value))

    model.add(layers.Flatten())
    model.add(layers.Dense(128 , activation=Modelparam.Activitation.layers))
    model.add(layers.Dropout(Modelparam.Dropout.Value))
    model.add(layers.Dense(Modelparam.MultiClass.num_classes , activation=Modelparam.Activitation.output))

    return model

#! training on multiple architecture
def trainingMultipleMethod(params, Data):
    Models, Hist = {}, {}
    for params.WhichExperiment.HardParams.Model.architectureType in ['U-Net' , 'CNN_Segmetnation']:
        architectureType = params.WhichExperiment.HardParams.Model.architectureType
        model = architecture(params)
        Models[architectureType], Hist[architectureType] = modelTrain(Data, params, model)

    return Models, Hist

#! applying the trained model on test data
def applyTestImageOnModel(model, Data, params, nameSubject, padding, ResultDir):
    pred = model.predict(Data.Image)
    score = model.evaluate(Data.Image, Data.Mask)

    num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes
    pred = np.transpose(pred,[1,2,0,3])[...,:num_classes-1]
    if len(pred.shape) == 3:
        pred= np.expand_dims(pred,axis=3)

    pred = smallFuncs.unPadding(pred, padding)
    Dice = np.zeros((num_classes-1,2))
    for cnt in range(num_classes-1):
        pred1N = np.squeeze(pred[...,cnt])
        origMsk1N = Data.OrigMask[...,cnt]
        Thresh = max( threshold_otsu(pred1N) ,0.2)  if len(np.unique(pred1N)) != 1 else 0
        # Thresh = 0.2

        pred1N = pred1N  > Thresh
        nucleusName, _ = smallFuncs.NucleiSelection(params.WhichExperiment.Nucleus.Index[cnt])
        dirSave = smallFuncs.mkDir(ResultDir + '/' + nameSubject)
        smallFuncs.saveImage(pred1N, Data.Affine, Data.Header, dirSave + '/' + nucleusName + '.nii.gz')
        Dice[cnt,:] = [ params.WhichExperiment.Nucleus.Index[cnt] , smallFuncs.Dice_Calculator(pred1N , origMsk1N) ]
    Dir_Dice = dirSave + '/Dice.txt' if params.WhichExperiment.HardParams.Model.MultiClass.mode else dirSave + '/Dice_' + nucleusName + '.txt'
    np.savetxt(Dir_Dice ,Dice)

    return Dice, pred, score
