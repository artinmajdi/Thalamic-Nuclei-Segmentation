import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras import models as kerasmodels
from keras import layers
# from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback # , TQDMNotebookCallback
import numpy as np
from skimage.filters import threshold_otsu
from Parameters import paramFunc
from otherFuncs import smallFuncs, datasets
from tqdm import tqdm
from time import time
import nibabel as nib
from scipy import ndimage
from shutil import copyfile
import pandas as pd

def check_Run(params, Data):

    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    model      = trainingExperiment(Data, params) if not params.preprocess.TestOnly else loadModel(params)
    prediction = testingExeriment(model, Data, params)

    if 'cascadeThalamusV1' in params.WhichExperiment.HardParams.Model.Idea and 1 in params.WhichExperiment.Nucleus.Index:         
        applyThalamusOnInput(params, prediction)

    return True

def loadModel(params):
    model = architecture(params)
    model.load_weights(params.directories.Train.Model + '/model_weights.h5')
    
    ModelParam = params.WhichExperiment.HardParams.Model
    model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
    return model
    
def testingExeriment(model, Data, params):

    # subject.NewCropInfo.PadSizeBackToOrig

    class prediction:
        Test = ''
        Train = ''

    def predictingTestSubject(model, Data, subject, params, nameSubject, padding, ResultDir):
        
        def postProcessing(params, subject, pred, origMsk1N, NucleiIndex):

            def binarizing(pred1N):
                Thresh = max( threshold_otsu(pred1N) ,0.2)  if len(np.unique(pred1N)) != 1 else 0
                # Thresh = 0.2
                return pred1N  > Thresh
            
            def cascade_paddingToOrigSize(params, im, subject):
                if 'cascadeThalamusV1' in params.WhichExperiment.HardParams.Model.Idea and 1 not in params.WhichExperiment.Nucleus.Index:
                    im = np.pad(im, subject.NewCropInfo.PadSizeBackToOrig, 'constant')
                return im
                        
            pred1N = binarizing( np.squeeze(pred) )                    
            pred1N_BtO = np.transpose(pred1N,params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse)
            pred1N_BtO = cascade_paddingToOrigSize(params, pred1N_BtO, subject)

            Dice = [ NucleiIndex , smallFuncs.Dice_Calculator(pred1N_BtO , origMsk1N) ]

            return pred1N_BtO, Dice

        def savingOutput(pred1N_BtO, params, ResultDir, nameSubject, NucleiIndex, Affine, Header):
            dirSave = smallFuncs.mkDir(ResultDir + '/' + nameSubject)  
            nucleusName, _ = smallFuncs.NucleiSelection(NucleiIndex)
            smallFuncs.saveImage( pred1N_BtO , Affine, Header, dirSave + '/' + nucleusName + '.nii.gz')
            return dirSave, nucleusName

        def loopOver_AllClasses_postProcessing(params, subject, pred, Data, num_classes, ResultDir, nameSubject):

            # TODO "pred1N_BtO" needs to concatenate for different classes

            Dice = np.zeros((num_classes-1,2))
            for cnt in range(num_classes-1):
                pred1N_BtO, Dice[cnt,:] = postProcessing(params, subject, pred[...,cnt], Data.OrigMask[...,cnt] , params.WhichExperiment.Nucleus.Index[cnt] )
                # pred1N = np.squeeze(pred[...,cnt])
                # origMsk1N = Data.OrigMask[...,cnt]
                # pred1N = binarizing(pred1N)                    
                # pred1N_BtO = np.transpose(pred1N,params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse)
                # pred1N_BtO = cascade_paddingToOrigSize(params, pred1N_BtO, subject)

                dirSave, nucleusName = savingOutput(pred1N_BtO, params, ResultDir, nameSubject, params.WhichExperiment.Nucleus.Index[cnt], Data.Affine, Data.Header)
                # dirSave = smallFuncs.mkDir(ResultDir + '/' + nameSubject)  
                # nucleusName, _ = smallFuncs.NucleiSelection(params.WhichExperiment.Nucleus.Index[cnt])
                # smallFuncs.saveImage( pred1N_BtO , Data.Affine, Data.Header, dirSave + '/' + nucleusName + '.nii.gz')

            Dir_Dice = dirSave + '/Dice.txt' if params.WhichExperiment.HardParams.Model.MultiClass.mode else dirSave + '/Dice_' + nucleusName + '.txt'
            np.savetxt(Dir_Dice ,Dice)
            return pred1N_BtO

        def applyPrediction(model, Data, num_classes, padding):
            pred = model.predict(Data.Image)
            # score = model.evaluate(Data.Image, Data.Mask)
            pred = pred[...,:num_classes-1]
            if len(pred.shape) == 3: pred = np.expand_dims(pred,axis=3)
            pred = np.transpose(pred,[1,2,0,3])
            if len(pred.shape) == 3: pred = np.expand_dims(pred,axis=3)
            pred = smallFuncs.unPadding(pred, padding)
            return pred


        num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes

        pred = applyPrediction(model, Data, num_classes, padding)

        pred1N_BtO = loopOver_AllClasses_postProcessing(params, subject, pred, Data, num_classes, ResultDir, nameSubject)

        return pred1N_BtO

    def loopOver_Predicting_TestSubjects(model, Data, params):
        prediction = {}
        ResultDir = params.directories.Test.Result
        for name in Data:            
            padding = params.directories.Test.Input.Subjects[name].Padding
            prediction[name] = predictingTestSubject(model, Data[name], params.directories.Test.Input.Subjects[name] , params, name, padding, ResultDir)

        return prediction

    def loopOver_Predicting_TrainSubjects(model, Data, params): 
        prediction = {}
        if (params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data) or ( 'cascadeThalamusV1' in params.WhichExperiment.HardParams.Model.Idea and 1 in params.WhichExperiment.Nucleus.Index):            
            ResultDir = smallFuncs.mkDir(params.directories.Test.Result + '/TrainData_Output')
            for name in Data:
                padding = params.directories.Train.Input.Subjects[name].Padding
                prediction[name] = predictingTestSubject(model, Data[name], params.directories.Train.Input.Subjects[name] , params, name, padding, ResultDir)

        return prediction

    prediction.Test  = loopOver_Predicting_TestSubjects(model, Data.Test, params)
    prediction.Train = loopOver_Predicting_TrainSubjects(model, Data.Train_ForTest, params)
    
    # smallFuncs.func_CropCoordinates(CropMask)
    
    return prediction

def trainingExperiment(Data, params):

    def saveTrainInfo(hist,a, params):
        hist.params['trainingTime'] = time() - a
        hist.params['InputDimensionsX'] = params.WhichExperiment.HardParams.Model.InputDimensions[0]
        hist.params['InputDimensionsY'] = params.WhichExperiment.HardParams.Model.InputDimensions[1]
        hist.params['num_Layers'] = params.WhichExperiment.HardParams.Model.num_Layers

        smallFuncs.saveReport(params.directories.Train.Model , 'hist_history' , hist.history , params.UserInfo['SaveReportMethod'])
        smallFuncs.saveReport(params.directories.Train.Model , 'hist_model'   , hist.model   , params.UserInfo['SaveReportMethod'])
        smallFuncs.saveReport(params.directories.Train.Model , 'hist_params'  , hist.params  , 'csv')
        
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
            hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Label), verbose=1, callbacks=[TQDMCallback()])

        smallFuncs.mkDir(params.directories.Train.Model)
        model.save(params.directories.Train.Model + '/model.h5', overwrite=True, include_optimizer=True )
        model.save_weights(params.directories.Train.Model + '/model_weights.h5', overwrite=True )
        if ModelParam.showHistory: print(hist.history)

        return model, hist

    a = time()
    smallFuncs.Saving_UserInfo(params.directories.Train.Model, params, params.UserInfo)
    model = architecture(params)
    model, hist = modelTrain(Data, params, model)

    saveTrainInfo(hist,a, params)
    return model

def applyThalamusOnInput(params, ThalamusMasks):

    params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)
    def ApplyThalamusMask(Thalamus_Mask, params, subject, nameSubject, mode):
            
        def dilateMask(mask, gapDilation):
            struc = ndimage.generate_binary_structure(3,2)
            struc = ndimage.iterate_structure(struc,gapDilation) 
            return ndimage.binary_dilation(Thalamus_Mask, structure=struc)
       
        def checkBordersOnBoundingBox(imFshape , BB , gapOnSlicingDimention):
            return [   [   np.max([BB[d][0]-gapOnSlicingDimention,0])  ,   np.min( [BB[d][1]+gapOnSlicingDimention,imFshape[d]])   ]  for d in range(3) ]

        def cropBoundingBoxes(params, subject, imFshape, Thalamus_Mask, Thalamus_Mask_Dilated):
            BB = smallFuncs.func_CropCoordinates(Thalamus_Mask)
            BB = checkBordersOnBoundingBox(imFshape , BB , params.WhichExperiment.Dataset.gapOnSlicingDimention)
            BBd  = smallFuncs.func_CropCoordinates(Thalamus_Mask_Dilated)

            np.savetxt(subject.Temp.address + '/BB.txt',BB,fmt='%d')
            np.savetxt(subject.Temp.address + '/BBd.txt',BBd,fmt='%d')
            return BB,BBd

        def apply_ThalamusMask_OnImage(Thalamus_Mask_Dilated, subject):

            if not os.path.isfile(subject.Temp.address + '/' + subject.ImageProcessed + '_BeforeThalamsMultiply.nii.gz'):
                copyfile(subject.address + '/' + subject.ImageProcessed + '.nii.gz' , subject.Temp.address + '/' + subject.ImageProcessed + '_BeforeThalamsMultiply.nii.gz')
            else:
                copyfile( subject.Temp.address + '/' + subject.ImageProcessed + '_BeforeThalamsMultiply.nii.gz' , subject.address + '/' + subject.ImageProcessed + '.nii.gz')
            
            imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
            im = imF.get_data()
            im[Thalamus_Mask_Dilated == 0] = 0
            smallFuncs.saveImage(im , imF.affine , imF.header , subject.address + '/' + subject.ImageProcessed + '.nii.gz')

        def saveNewCrop(BB,BBd, imFshape, subject):

            newCrop0 = newCrop1 = newCrop2 = np.zeros(imFshape)
            newCrop0[ BB[0][0] :BB[0][-1]   ,  BBd[1][0]:BBd[1][-1]  ,  BBd[2][0]:BBd[2][-1] ] = 1
            newCrop1[ BBd[0][0]:BBd[0][-1]  ,  BB[1][0] :BB[1][-1]   ,  BBd[2][0]:BBd[2][-1] ] = 1
            newCrop2[ BBd[0][0]:BBd[0][-1]  ,  BBd[1][0]:BBd[1][-1]  ,  BB[2][0] :BB[2][-1]  ] = 1

            smallFuncs.saveImage(newCrop0 , imF.affine , imF.header , subject.Temp.address + '/CropMask_ThCascade_sliceDim0.nii.gz')
            smallFuncs.saveImage(newCrop1 , imF.affine , imF.header , subject.Temp.address + '/CropMask_ThCascade_sliceDim1.nii.gz')
            smallFuncs.saveImage(newCrop2 , imF.affine , imF.header , subject.Temp.address + '/CropMask_ThCascade_sliceDim2.nii.gz')

        # if not os.path.isfile(subject.Temp.address + '/BB.txt'):

        imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')

        # TODO Feb 7 check of dilated values are 0 and 1
        Thalamus_Mask_Dilated = dilateMask( Thalamus_Mask, params.WhichExperiment.Dataset.gapDilation )
        BB,BBd = cropBoundingBoxes(params, subject, imF.shape, Thalamus_Mask, Thalamus_Mask_Dilated)

        imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
        apply_ThalamusMask_OnImage(Thalamus_Mask_Dilated, subject)
        saveNewCrop(BB,BBd, imF.shape, subject)
    
    def loopOverSubjects(params, ThalamusMasks, mode):
        Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects
        for sj in tqdm(Subjects ,desc='applying Thalamus for cascade method: ' + mode):           
            ApplyThalamusMask(ThalamusMasks[sj] , params, Subjects[sj], sj, 'train') 

    loopOverSubjects(params, ThalamusMasks.Test, 'test')

    if not params.preprocess.TestOnly or params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data:  
        loopOverSubjects(params, ThalamusMasks.Train, 'train')

# ! U-Net Architecture
def architecture(params):

    def UNet(Modelparam):

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

    # params.WhichExperiment.HardParams.Model.imageInfo = Data.Info
    ModelParam = params.WhichExperiment.HardParams.Model
    if 'U-Net' in ModelParam.architectureType:
        model = UNet(ModelParam)

    elif 'CNN_Segmetnation' in ModelParam.architectureType:
        model = CNN_Segmetnation(ModelParam)

    elif 'CNN_Classifier' in ModelParam.architectureType:
        model = CNN_Segmetnation(ModelParam)

    # model.summary()

    # ModelParam = params.WhichExperiment.HardParams.Model
    # model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
    return model

def CNN_Segmetnation(Modelparam):
    inputs = layers.Input( (Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1) )
    conv = inputs

    for nL in range(Modelparam.num_Layers -1):
        conv = layers.Conv2D(filters=64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)
        conv = layers.Dropout(Modelparam.Dropout.Value)(conv)

    final  = layers.Conv2D(filters=Modelparam.MultiClass.num_classes, kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)

    model = kerasmodels.Model(inputs=[inputs], outputs=[final])

    return model

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
    
