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
from preprocess import croppingA
from tqdm import tqdm
from time import time
import nibabel as nib
from scipy import ndimage
from shutil import copyfile
import pandas as pd
import mat4py
import pickle
from skimage import measure

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

    class prediction:
        Test = ''
        Train = ''

    def predictingTestSubject(DataSubj, subject, ResultDir):
        
        def postProcessing(pred1Class, origMsk1N, NucleiIndex):

            def binarizing(pred1N):
                Thresh = max( threshold_otsu(pred1N) ,0.2)  if len(np.unique(pred1N)) != 1 else 0
                # Thresh = 0.2
                return pred1N  > Thresh
            
            def cascade_paddingToOrigSize(im):
                if 'cascadeThalamusV1' in params.WhichExperiment.HardParams.Model.Idea and 1 not in params.WhichExperiment.Nucleus.Index:
                    im = np.pad(im, subject.NewCropInfo.PadSizeBackToOrig, 'constant')
                    # Padding2, crd = paddingNegativeFix(im.shape, Padding2)          
                    # im = im[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]
                return im
                        
            pred1N = binarizing( np.squeeze(pred1Class) )                    
            
            pred1N_origShape = cascade_paddingToOrigSize(pred1N)
            pred1N_origShape = np.transpose(pred1N_origShape,params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse)

            Dice = [ NucleiIndex , smallFuncs.Dice_Calculator(pred1N_origShape , origMsk1N) ]

            return pred1N_origShape, Dice

        def savingOutput(pred1N_BtO, NucleiIndex):
            dirSave = smallFuncs.mkDir(ResultDir + '/' + subject.subjectName)  
            nucleusName, _ , _ = smallFuncs.NucleiSelection(NucleiIndex)
            smallFuncs.saveImage( pred1N_BtO , DataSubj.Affine, DataSubj.Header, dirSave + '/' + nucleusName + '.nii.gz')
            return dirSave, nucleusName

        def applyPrediction():

            num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes
            def loopOver_AllClasses_postProcessing(pred):

                # TODO "pred1N_BtO" needs to concatenate for different classes

                Dice = np.zeros((num_classes-1,2))
                for cnt in range(num_classes-1):

                    pred1N_BtO, Dice[cnt,:] = postProcessing(pred[...,cnt], DataSubj.OrigMask[...,cnt] , params.WhichExperiment.Nucleus.Index[cnt] )

                    dirSave, nucleusName = savingOutput(pred1N_BtO, params.WhichExperiment.Nucleus.Index[cnt])

                Dir_Dice = dirSave + '/Dice.txt' if params.WhichExperiment.HardParams.Model.MultiClass.mode else dirSave + '/Dice_' + nucleusName + '.txt'
                np.savetxt(Dir_Dice ,Dice)
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

            pred = model.predict(DataSubj.Image)
            # score = model.evaluate(DataSubj.Image, DataSubj.Mask)
            pred = pred[...,:num_classes-1]
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
        for name in DataTest:            
            prediction[name] = predictingTestSubject(DataTest[name], params.directories.Test.Input.Subjects[name] , ResultDir)

        return prediction

    def loopOver_Predicting_TrainSubjects(DataTrain): 
        prediction = {}
        if (params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data) or ( 'cascadeThalamusV1' in params.WhichExperiment.HardParams.Model.Idea and 1 in params.WhichExperiment.Nucleus.Index):            
            ResultDir = smallFuncs.mkDir(params.directories.Test.Result + '/TrainData_Output')
            for name in DataTrain:
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
        hist.params['trainingTime'] = time() - a
        hist.params['InputDimensionsX'] = params.WhichExperiment.HardParams.Model.InputDimensions[0]
        hist.params['InputDimensionsY'] = params.WhichExperiment.HardParams.Model.InputDimensions[1]
        hist.params['num_Layers'] = params.WhichExperiment.HardParams.Model.num_Layers

        saveReport(params.directories.Train.Model , 'hist_history' , hist.history , params.UserInfo['SaveReportMethod'])
        saveReport(params.directories.Train.Model , 'hist_model'   , hist.model   , params.UserInfo['SaveReportMethod'])
        saveReport(params.directories.Train.Model , 'hist_params'  , hist.params  , 'csv')
        
    def modelTrain_Unet(Data, params, model):
        ModelParam = params.WhichExperiment.HardParams.Model


        if params.WhichExperiment.Nucleus.Index[0] != 1 and params.WhichExperiment.HardParams.Model.InitializeFromThalamus and os.path.exists(params.directories.Train.Model_Thalamus + '/model_weights.h5'):
            model.load_weights(params.directories.Train.Model_Thalamus + '/model_weights.h5')
        elif params.WhichExperiment.HardParams.Model.InitializeFromOlderModel and os.path.exists(params.directories.Train.Model + '/model_weights.h5'):
            model.load_weights(params.directories.Train.Model + '/model_weights.h5')

        model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)

        # if the shuffle argument in model.fit is set to True (which is the default), the training data will be randomly shuffled at each epoch.
        if params.WhichExperiment.Dataset.Validation.fromKeras:
            hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs, shuffle=True, validation_split=params.WhichExperiment.Dataset.Validation.percentage, verbose=1) # , callbacks=[TQDMCallback()])
        else:
            hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Label), verbose=1) # , callbacks=[TQDMCallback()])

        smallFuncs.mkDir(params.directories.Train.Model)
        model.save(params.directories.Train.Model + '/model.h5', overwrite=True, include_optimizer=True )
        model.save_weights(params.directories.Train.Model + '/model_weights.h5', overwrite=True )
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
        UserInfo['num_Layers']      = params.WhichExperiment.HardParams.Model.num_Layers

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

def applyThalamusOnInput(params, ThalamusMasks):

    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    def ApplyThalamusMask(Thalamus_Mask, subject, mode):
                  
        def cropBoundingBoxes(Thalamus_Mask):

            def dilateMask(mask, gapDilation):
                struc = ndimage.generate_binary_structure(3,2)
                struc = ndimage.iterate_structure(struc,gapDilation) 
                return ndimage.binary_dilation(mask, structure=struc)
                
            def checkBordersOnBoundingBox(imFshape , BB , gapOnSlicingDimention):
                return [   [   np.max([BB[d][0]-gapOnSlicingDimention,0])  ,   np.min( [BB[d][1]+gapOnSlicingDimention,imFshape[d]])   ]  for d in range(3) ]

            def findBoundingBox(Thalamus_Mask):
                objects = measure.regionprops(measure.label(Thalamus_Mask))
                area = []
                for obj in objects: area = np.append(area, obj.area)

                Ix = np.argsort(area)

                L = len(Thalamus_Mask.shape)
                bbox = objects[ Ix[-1] ].bbox
                BB = [ [bbox[d] , bbox[L + d] ] for d in range(L)]
                                
                return BB

            # Thalamus_Mask_Dilated = dilateMask( Thalamus_Mask, params.WhichExperiment.Dataset.gapDilation )
            imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')

            BB = findBoundingBox(Thalamus_Mask)
            # BB = croppingA.func_CropCoordinates(Thalamus_Mask)

            BB = checkBordersOnBoundingBox(imF.shape , BB , params.WhichExperiment.Dataset.gapOnSlicingDimention)
            # BBd  = croppingA.func_CropCoordinates(Thalamus_Mask_Dilated)
            BBd = [  [BB[ii][0] - 5 , BB[ii][1] + 5] for ii in range(len(BB))]
            BBd = checkBordersOnBoundingBox(imF.shape , BBd , 0)

            dirr = params.directories.Test.Result 
            if 'train' in mode: dirr += '/TrainData_Output'
                           
            np.savetxt(dirr + '/' + subject.subjectName + '/BB.txt',BB,fmt='%d')
            np.savetxt(dirr + '/' + subject.subjectName + '/BBd.txt',BBd,fmt='%d')
            
        # def apply_ThalamusMask_OnImage(Thalamus_Mask_Dilated, subject):
        #     imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
        #     im = imF.get_data()
        #     im[Thalamus_Mask_Dilated == 0] = 0
        #     # smallFuncs.saveImage(im , imF.affine , imF.header , subject.address + '/' + subject.ImageProcessed + '.nii.gz')

   
        cropBoundingBoxes(Thalamus_Mask)

        # imF = apply_ThalamusMask_OnImage(Thalamus_Mask_Dilated, subject)
    
    def loopOverSubjects(ThalamusMasks, mode):
        Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects
        for sj in tqdm(Subjects ,desc='applying Thalamus for cascade method: ' + mode):         
            ApplyThalamusMask(ThalamusMasks[sj] , Subjects[sj] , mode) 

    loopOverSubjects(ThalamusMasks.Test, 'test')

    if not params.preprocess.TestOnly or params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data:  
        loopOverSubjects(ThalamusMasks.Train, 'train')

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
        
    # params.WhichExperiment.HardParams.Model.imageInfo = Data.Info
    ModelParam = params.WhichExperiment.HardParams.Model
    if 'U-Net' in ModelParam.architectureType:
        model = UNet(ModelParam)

    elif 'FCN_Cropping' in ModelParam.architectureType:
        model = CNN_Segmetnation(ModelParam)

    elif 'CNN_Classifier' in ModelParam.architectureType:
        model = CNN_Classifier(ModelParam)

    # model.summary()

    # ModelParam = params.WhichExperiment.HardParams.Model
    # model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
    return model

