import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras import models as kerasmodels
from keras import layers
# from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback # , TQDMNotebookCallback
import numpy as np
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import preprocess.croppingA as croppingA
from tqdm import tqdm
from time import time
import nibabel as nib
from scipy import ndimage
from shutil import copyfile
import pandas as pd
import mat4py
import pickle
import skimage
import keras
import keras.preprocessing
from keras.utils import multi_gpu_model, multi_gpu_utils
import h5py

# keras.preprocessing.utils.multi_gpu_model
def check_Run(params, Data):

    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    model      = trainingExperiment(Data, params) if not params.preprocess.TestOnly else loadModel(params)
    prediction = testingExeriment(model, Data, params)

    if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and (int(params.WhichExperiment.Nucleus.Index[0]) == 1):
        savePreFinalStageBBs(params, prediction)

    return True

def loadModel(params):
    # model = architecture(params)
    # model.load_weights(params.directories.Train.Model + '/model_weights.h5')

    # with open(params.directories.Train.Model + '/model' + tagTF + '.json', 'r') as json_file:
    #     model2 = keras.models.model_from_json(json_file.read())

    # model2.load_weights("model.h5")
    
    model = kerasmodels.load_model(params.directories.Train.Model + '/model.h5')

    ModelParam = params.WhichExperiment.HardParams.Model
    model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
    return model

def testingExeriment(model, Data, params):

    class prediction:
        Test = ''
        Train = ''

    def predictingTestSubject(DataSubj, subject, ResultDir):

        def postProcessing(pred1Class, origMsk1N, NucleiIndex):

            def binarizing(pred1N):
                Thresh = max( skimage.filters.threshold_otsu(pred1N) ,0.2)  if len(np.unique(pred1N)) != 1 else 0
                return pred1N  > Thresh

            def cascade_paddingToOrigSize(im):
                if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and 1 not in params.WhichExperiment.Nucleus.Index:
                    im = np.pad(im, subject.NewCropInfo.PadSizeBackToOrig, 'constant')
                    # Padding2, crd = paddingNegativeFix(im.shape, Padding2)
                    # im = im[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]
                return im

            pred1N = binarizing( np.squeeze(pred1Class) )

            pred1N_origShape = cascade_paddingToOrigSize(pred1N)
            pred1N_origShape = np.transpose(pred1N_origShape,params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse)

            Dice = [ NucleiIndex , smallFuncs.Dice_Calculator(pred1N_origShape , binarizing(origMsk1N)) ]

            return pred1N_origShape, Dice

        def savingOutput(pred1N_BtO, NucleiIndex):
            dirSave = smallFuncs.mkDir(ResultDir + '/' + subject.subjectName)
            nucleusName, _ , _ = smallFuncs.NucleiSelection(NucleiIndex)

            smallFuncs.saveImage( pred1N_BtO , DataSubj.Affine, DataSubj.Header, dirSave + '/' + nucleusName + '.nii.gz')
            return dirSave, nucleusName

        def applyPrediction():

            num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes
            if not params.WhichExperiment.HardParams.Model.Method.havingBackGround_AsExtraDimension:
                num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes + 1

            def loopOver_AllClasses_postProcessing(pred):

                # TODO "pred1N_BtO" needs to concatenate for different classes
                Dice = np.zeros((num_classes-1,2))
                for cnt in range(num_classes-1):

                    pred1N_BtO, Dice[cnt,:] = postProcessing(pred[...,cnt], DataSubj.OrigMask[...,cnt] , params.WhichExperiment.Nucleus.Index[cnt] )

                    dirSave, nucleusName = savingOutput(pred1N_BtO, params.WhichExperiment.Nucleus.Index[cnt])

                Dir_Dice = dirSave + '/Dice.txt' if params.WhichExperiment.HardParams.Model.MultiClass.mode else dirSave + '/Dice_' + nucleusName + '.txt'
                np.savetxt(Dir_Dice ,Dice,fmt='%1.1f %1.4f')
                return pred1N_BtO

            def unPadding(im , pad):
                sz = im.shape
                if np.min(pad) < 0:
                    pad, crd = datasets.paddingNegativeFix(sz, pad)
                    for ix in range(3):
                         crd[ix,1] = 0  if crd[ix,1] == sz[ix] else -crd[ix,1]

                    # if len(crd) == 3: crd = np.append(crd,[0,0],axs=1)
                    crd = tuple([tuple(x) for x in crd])

                    im = im[pad[0][0]:sz[0]-pad[0][1] , pad[1][0]:sz[1]-pad[1][1] , pad[2][0]:sz[2]-pad[2][1],:]
                    im = np.pad( im , crd , 'constant')
                else:
                    im = im[pad[0][0]:sz[0]-pad[0][1] , pad[1][0]:sz[1]-pad[1][1] , pad[2][0]:sz[2]-pad[2][1],:]

                return im

            im = DataSubj.Image.copy()
            if params.WhichExperiment.Dataset.slicingInfo.slicingDim == 0:
                class cropSD0:
                    crop = int(im.shape[0]/2+5)
                    padBackToOrig = im.shape[0] - int(im.shape[0]/2+5)
                im = im[:cropSD0.crop,...]

            predF = model.predict(im)

            if params.WhichExperiment.Dataset.slicingInfo.slicingDim == 0:
                predF = np.pad(predF,((0,cropSD0.padBackToOrig),(0,0),(0,0),(0,0)),mode='constant')


            # score = model.evaluate(DataSubj.Image, DataSubj.Mask)
            pred = predF[...,:num_classes-1]

            if len(pred.shape) == 3: pred = np.expand_dims(pred,axis=3)
            pred = np.transpose(pred,[1,2,0,3])
            if len(pred.shape) == 3: pred = np.expand_dims(pred,axis=3)

            # paddingTp = [ subject.Padding[p] for p in params.WhichExperiment.Dataset.slicingInfo.slicingOrder]
            # if len(paddingTp) == 3: paddingTp.append((0,0))
            pred = unPadding(pred, subject.Padding)
            return loopOver_AllClasses_postProcessing(pred)

        return applyPrediction()

    def loopOver_Predicting_TestSubjects(DataTest):
        prediction = {}
        ResultDir = params.directories.Test.Result
        for name in tqdm(DataTest,desc='predicting test subjects'):
            prediction[name] = predictingTestSubject(DataTest[name], params.directories.Test.Input.Subjects[name] , ResultDir)

        return prediction

    def loopOver_Predicting_TrainSubjects(DataTrain):
        prediction = {}
        if (params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data) or ( 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and (int(params.WhichExperiment.Nucleus.Index[0]) == 1)):
            ResultDir = smallFuncs.mkDir(params.directories.Test.Result + '/TrainData_Output')
            for name in tqdm(DataTrain,desc='predicting train subjects'):
                prediction[name] = predictingTestSubject(DataTrain[name], params.directories.Train.Input.Subjects[name] , ResultDir)

        return prediction

    prediction.Test  = loopOver_Predicting_TestSubjects(Data.Test)
    prediction.Train = loopOver_Predicting_TrainSubjects(Data.Train_ForTest)

    return prediction

def trainingExperiment(Data, params):

    def saveReport(DirSave, name , data, method):

        def savePickle(Dir, data):
            f = open(Dir,"wb")
            pickle.dump(data,f)
            f.close()

        if 'pickle' in method: savePickle(DirSave + '/' + name + '.pkl', data)
        elif 'mat'  in method: mat4py.savemat(DirSave + '/' + name + '.mat', data)
        elif 'csv'  in method: pd.DataFrame(data=data,columns=list(data.keys())).to_csv( DirSave + '/' + name + '.csv')

    def saveTrainInfo(hist,a, params):

        hist.params['trainingTime']     = time() - a
        hist.params['InputDimensionsX'] = params.WhichExperiment.HardParams.Model.InputDimensions[0]
        hist.params['InputDimensionsY'] = params.WhichExperiment.HardParams.Model.InputDimensions[1]
        hist.params['InputDimensionsZ'] = params.WhichExperiment.HardParams.Model.InputDimensions[2]
        hist.params['num_Layers']       = params.WhichExperiment.HardParams.Model.num_Layers

        saveReport(params.directories.Train.Model , 'hist_history' , hist.history , params.UserInfo['SaveReportMethod'])
        saveReport(params.directories.Train.Model , 'hist_model'   , hist.model   , params.UserInfo['SaveReportMethod'])
        saveReport(params.directories.Train.Model , 'hist_params'  , hist.params  , params.UserInfo['SaveReportMethod'])

    def modelTrain_Unet(Data, params, model):
        ModelParam = params.WhichExperiment.HardParams.Model

        def saveModel_h5(model2, params):       
            smallFuncs.mkDir(params.directories.Train.Model)
            tagTF = '_TF' if params.WhichExperiment.HardParams.Model.Transfer_Learning.Mode else ''
            model2.save(params.directories.Train.Model + '/model' + tagTF + '.h5', overwrite=True, include_optimizer=False )
            model2.save_weights(params.directories.Train.Model + '/model_weights' + tagTF + '.h5', overwrite=True )

            with open(params.directories.Train.Model + '/model' + tagTF + '.json', "w") as json_file:
                json_file.write(model2.to_json())

            keras.utils.plot_model(model,to_file=params.directories.Train.Model+'/Architecture.png',show_layer_names=True,show_shapes=True)
            
        def modelInitialize(model2):

            # with open(params.directories.Train.Model + '/model' + tagTF + '.json', 'r') as json_file:
            #     model2 = keras.models.model_from_json(json_file.read())

            # model2.load_weights("model.h5")
            try:
                if params.WhichExperiment.Nucleus.Index[0] != 1 and params.WhichExperiment.HardParams.Model.InitializeFromThalamus and os.path.exists(params.directories.Train.Model_Thalamus + '/model_weights.h5'):
                    model2.load_weights(params.directories.Train.Model_Thalamus + '/model_weights.h5')
                elif params.WhichExperiment.HardParams.Model.InitializeFromOlderModel and os.path.exists(params.directories.Train.Model + '/model_weights.h5'):
                    model2.load_weights(params.directories.Train.Model + '/model_weights.h5')
                elif params.WhichExperiment.HardParams.Model.Initialize_From_3T and os.path.exists(params.directories.Train.Model_3T + '/model_weights.h5'):
                    model2.load_weights(params.directories.Train.Model_3T + '/model_weights.h5')
                    print('Model_3T' , params.directories.Train.Model_3T)

                if params.WhichExperiment.HardParams.Model.Transfer_Learning.Mode:
                    model2.load_weights(params.directories.Train.Model + '/model_weights.h5')
            except: 
                print('loading Initial Weights Failed')
            return model2

        def modelFit(params):

            def func_modelParams():
                batch_size       = params.WhichExperiment.HardParams.Model.batch_size
                epochs           = params.WhichExperiment.HardParams.Model.epochs 
                valSplit_Per = params.WhichExperiment.Dataset.Validation.percentage
                verbose          = params.WhichExperiment.HardParams.Model.verbose
                ManualDataGenerator = params.WhichExperiment.HardParams.Model.ManualDataGenerator
                Validation_fromKeras = params.WhichExperiment.Dataset.Validation.fromKeras
                return batch_size, epochs, valSplit_Per, verbose , ManualDataGenerator , Validation_fromKeras
            batch_size, epochs, valSplit_Per, verbose , ManualDataGenerator, Validation_fromKeras = func_modelParams()

            def RunGenerator(params):
                f = h5py.File(params.directories.Test.Result + '/Data.hdf5','r')
                    
                infoDG = {'dim': tuple(f['Train/Image'].shape[1:3]),'batch_size': ModelParam.batch_size,
                        'n_classes': ModelParam.MultiClass.num_classes, 'n_channels': f['Train/Image'].shape[3]}
                                
                training_generator   = DataGenerator( f['Train'], **infoDG )
                validation_generator = DataGenerator( f['Validation'], **infoDG )

                hist = model.fit_generator(generator=training_generator, validation_data=validation_generator, verbose=1)   # , use_multiprocessing=True, workers=20
                f.close()
                return hist

            if ManualDataGenerator: hist = RunGenerator(params)
            else:
                if Validation_fromKeras: hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=valSplit_Per, verbose=verbose) # , callbacks=[TQDMCallback()])
                else:                    hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Mask), verbose=verbose) # , callbacks=[TQDMCallback()])        
                
            return hist

        model = modelInitialize(model)

        if len(params.WhichExperiment.HardParams.Machine.GPU_Index) > 1:   model = multi_gpu_model(model)

        model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)

        hist = modelFit(params)

        saveModel_h5(model, params)
        if ModelParam.showHistory: print(hist.history)

        return model, hist

    def modelTrain_Cropping(Data, params, model):
        ModelParam = params.WhichExperiment.HardParams.Model
        model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)

        if params.WhichExperiment.Dataset.Validation.fromKeras:
            hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs, shuffle=True, validation_split=params.WhichExperiment.Dataset.Validation.percentage, verbose=1) # , callbacks=[TQDMCallback()])
        else:
            hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Label), verbose=1) # , callbacks=[TQDMCallback()])

        smallFuncs.mkDir(params.directories.Train.Model)
        model.save(params.directories.Train.Model + '/model.h5', overwrite=True, include_optimizer=True )
        model.save_weights(params.directories.Train.Model + '/model_weights.h5', overwrite=True )
        if ModelParam.showHistory: print(hist.history)

        return model, hist

    def Saving_UserInfo(DirSave, params, UserInfo):

        UserInfo['InputDimensions'] = str(params.WhichExperiment.HardParams.Model.InputDimensions)
        UserInfo['simulation'].num_Layers      = params.WhichExperiment.HardParams.Model.num_Layers

        saveReport(DirSave, 'UserInfo', UserInfo , UserInfo['SaveReportMethod'])

    a = time()
    Saving_UserInfo(params.directories.Train.Model, params, params.UserInfo)
    model = architecture(params)

    if 'U-Net' in params.WhichExperiment.HardParams.Model.architectureType:
        model, hist = modelTrain_Unet(Data, params, model)
        saveTrainInfo(hist,a, params)
    elif 'FCN_Cropping':
        model, hist = modelTrain_Unet(Data, params, model)

    return model

def savePreFinalStageBBs(params, CascadePreStageMasks):

    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    def ApplyPreFinalStageMask(PreStageMask, subject, mode):

        def cropBoundingBoxes(PreStageMask):

            def checkBordersOnBoundingBox(imFshape , BB , gapOnSlicingDimention):
                return [   [   np.max([BB[d][0]-gapOnSlicingDimention,0])  ,   np.min( [BB[d][1]+gapOnSlicingDimention,imFshape[d]])   ]  for d in range(3) ]

            # Thalamus_Mask_Dilated = dilateMask( PreStageMask, params.WhichExperiment.Dataset.gapDilation )
            imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')

            BB = smallFuncs.findBoundingBox(PreStageMask)
            BB = checkBordersOnBoundingBox(imF.shape , BB , params.WhichExperiment.Dataset.gapOnSlicingDimention)
            gapDilation = params.WhichExperiment.Dataset.gapDilation
            BBd = [  [BB[ii][0] - gapDilation , BB[ii][1] + gapDilation] for ii in range(len(BB))]
            BBd = checkBordersOnBoundingBox(imF.shape , BBd , 0)

            dirr = params.directories.Test.Result
            if 'train' in mode: dirr += '/TrainData_Output'

            np.savetxt(dirr + '/' + subject.subjectName + '/BB_' + params.WhichExperiment.Nucleus.name + '.txt',np.concatenate((BB,BBd),axis=1),fmt='%d')

        cropBoundingBoxes(PreStageMask)

    def loopOverSubjects(CascadePreStageMasks, mode):
        Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects
        string = 'Thalamus' if 1 in params.WhichExperiment.Nucleus.Index else 'stage2 ' + params.WhichExperiment.Nucleus.name
        for sj in tqdm(Subjects ,desc='applying ' + string + ' for cascade method: ' + mode):
            ApplyPreFinalStageMask(CascadePreStageMasks[sj] , Subjects[sj] , mode)

    loopOverSubjects(CascadePreStageMasks.Test, 'test')
    loopOverSubjects(CascadePreStageMasks.Train, 'train')

def architecture(params):

    
    def UNet(Modelparam):

        NumberFM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num

        def Unet_sublayer_Contracting(inputs, nL, Modelparam):

            Trainable = False if Modelparam.Transfer_Learning.Mode and nL in ModelParam.Transfer_Learning.FrozenLayers else True
            if Modelparam.Layer_Params.batchNormalization:  inputs = layers.BatchNormalization()(inputs)
            conv = layers.Conv2D(NumberFM*(2**nL), kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.conv, padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.layers, trainable=Trainable)(inputs)
            conv = layers.Conv2D(NumberFM*(2**nL), kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.conv, padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.layers, trainable=Trainable)(conv)
            pool = layers.MaxPooling2D( pool_size=Modelparam.Layer_Params.MaxPooling.pool_size)(conv)
            if Modelparam.Layer_Params.Dropout.Mode and Trainable: pool = layers.Dropout(Modelparam.Layer_Params.Dropout.Value)(pool)
            return pool, conv

        def Unet_sublayer_Expanding(inputs, nL, Modelparam, contractingInfo):

            Trainable = False if Modelparam.Transfer_Learning.Mode and nL in ModelParam.Transfer_Learning.FrozenLayers else True
            if Modelparam.Layer_Params.batchNormalization:  inputs = layers.BatchNormalization()(inputs)
            UP = layers.Conv2DTranspose(NumberFM*(2**nL), kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.convTranspose, strides=(2,2), padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.layers, trainable=Trainable)(inputs)
            UP = layers.merge.concatenate([UP,contractingInfo[nL+1]],axis=3)
            conv = layers.Conv2D(NumberFM*(2**nL), kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.conv, padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.layers, trainable=Trainable)(UP)
            conv = layers.Conv2D(NumberFM*(2**nL), kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.conv, padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.layers, trainable=Trainable)(conv)
            if Modelparam.Layer_Params.Dropout.Mode and Trainable: conv = layers.Dropout(Modelparam.Layer_Params.Dropout.Value)(conv)
            return conv

        def Unet_MiddleLayer(WeightBiases, nL, Modelparam):
            Trainable = False if Modelparam.Transfer_Learning.Mode and nL in ModelParam.Transfer_Learning.FrozenLayers else True
            WeightBiases = layers.Conv2D(NumberFM*(2**nL), kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.conv, padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.layers, trainable=Trainable)(WeightBiases)
            WeightBiases = layers.Conv2D(NumberFM*(2**nL), kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.conv, padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.layers, trainable=Trainable)(WeightBiases)
            if Modelparam.Layer_Params.Dropout.Mode and Trainable: WeightBiases = layers.Dropout(Modelparam.Layer_Params.Dropout.Value)(WeightBiases)
            return WeightBiases

        dim = params.WhichExperiment.HardParams.Model.Method.InputImage2Dvs3D
        inputs = layers.Input( tuple(Modelparam.InputDimensions[:dim]) + (1,) )
        
        # ! contracting layer
        ConvOutputs = {}
        WeightBiases = inputs
        for nL in range(Modelparam.num_Layers -1):
            WeightBiases, ConvOutputs[nL+1] = Unet_sublayer_Contracting(WeightBiases, nL, Modelparam)

        # ! middle layer
        WeightBiases = Unet_MiddleLayer(WeightBiases , Modelparam.num_Layers-1 , Modelparam)

        # ! expanding layer
        for nL in reversed(range(Modelparam.num_Layers -1)):
            WeightBiases = Unet_sublayer_Expanding(WeightBiases, nL, Modelparam, ConvOutputs)

        # ! Final layer
        final = layers.Conv2D(Modelparam.MultiClass.num_classes, kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.output, padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.output)(WeightBiases)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    def CNN_Classifier(Modelparam):
        dim = Modelparam.Method.InputImage2Dvs3D
        model = kerasmodels.Sequential()
        model.add(layers.Conv2D(filters=16, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape= tuple(Modelparam.InputDimensions[:dim]) + (1,)  ))
        model.add(layers.MaxPooling2D(pool_size=Modelparam.Layer_Params.MaxPooling.pool_size))
        model.add(layers.Dropout(Modelparam.Layer_Params.Dropout.Value))

        model.add(layers.Conv2D(filters=8, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape=tuple(Modelparam.InputDimensions[:dim]) + (1,) ))
        model.add(layers.MaxPooling2D(pool_size=Modelparam.Layer_Params.MaxPooling.pool_size))
        model.add(layers.Dropout(Modelparam.Layer_Params.Dropout.Value))

        model.add(layers.Flatten())
        model.add(layers.Dense(128 , activation=Modelparam.Layer_Params.Activitation.layers))
        model.add(layers.Dropout(Modelparam.Layer_Params.Dropout.Value))
        model.add(layers.Dense(Modelparam.MultiClass.num_classes , activation=Modelparam.Layer_Params.Activitation.output))

        return model

    def FCN(Modelparam):
        dim = Modelparam.Method.InputImage2Dvs3D
        inputs = layers.Input( tuple(Modelparam.InputDimensions[:dim]) + (1,) )

        conv = inputs
        for nL in range(Modelparam.num_Layers -1):
            Trainable = False if Modelparam.Transfer_Learning.Mode and nL in ModelParam.Transfer_Learning.FrozenLayers else True
            conv = layers.Conv3D(filters=ModelParam.Layer_Params.FirstLayer_FeatureMap_Num*(2**nL), kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.conv, padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.layers, trainable=Trainable)(conv)
            if Modelparam.Layer_Params.Dropout.Mode and Trainable: conv = layers.Dropout(Modelparam.Layer_Params.Dropout)(conv)

        final  = layers.Conv3D(filters=Modelparam.MultiClass.num_classes, kernel_size=Modelparam.Layer_Params.ConvLayer.Kernel_size.output, padding=Modelparam.Layer_Params.ConvLayer.padding, activation=Modelparam.Layer_Params.Activitation.output)(conv)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    # params.WhichExperiment.HardParams.Model.imageInfo = Data.Info
    ModelParam = params.WhichExperiment.HardParams.Model
    if 'U-Net' in ModelParam.architectureType:
        model = UNet(ModelParam)

    elif 'CNN_Classifier' in ModelParam.architectureType:
        model = CNN_Classifier(ModelParam)

    model.summary()

    return model

class DataGenerator(keras.utils.Sequence):
    def __init__(self, h, batch_size=100, dim=(32,32), n_channels=1, n_classes=2):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.h = h
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.h['Image'].shape[0] / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[ index*self.batch_size : (index+1)*self.batch_size ]
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        self.indexes = np.arange(self.h['Image'].shape[0])
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        return self.h['Image'][list_IDs_temp,...] , self.h['Mask'][list_IDs_temp,...]
