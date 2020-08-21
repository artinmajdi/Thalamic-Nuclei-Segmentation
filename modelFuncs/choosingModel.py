import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras import models as kerasmodels
import numpy as np
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
from tqdm import tqdm
import nibabel as nib
from scipy import ndimage
import mat4py
import pickle
import keras
from time import time
from keras.utils import multi_gpu_model
import h5py
import keras.layers as KLayers
import modelFuncs.LossFunction as LossFunction
from skimage.transform import AffineTransform , warp

def func_class_weights(weighted_Mode, Mask):
    sz = Mask.shape
    NUM_CLASSES = sz[3]
    class_weights = np.ones(NUM_CLASSES)

    #! equal weights
    if weighted_Mode:
        for ix in range(NUM_CLASSES):                                        
            TRUE_Count = len(np.where(Mask[...,ix] > 0.5)[0])
            NUM_SAMPLES = np.prod(sz[:3])
            class_weights[ix] = NUM_SAMPLES / (NUM_CLASSES*TRUE_Count)
    
    class_weights = class_weights/np.sum(class_weights)
    print('class_weights' , class_weights)
    # # ! zero weight for foreground
    # for i in [0,2,5,7]: class_weights[i] = 0
    # # class_weights = class_weight.compute_class_weight('balanced',classes,y_train)
    return class_weights
    
def check_Run(params, Data):

    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    
    if params.WhichExperiment.TestOnly or params.UserInfo['thalamic_side'].active_side == 'right':
        model = loadModel(params)
    else:
        model = trainingExperiment(Data, params)
    
    prediction = testingExeriment(model, Data, params)

    method = params.WhichExperiment.HardParams.Model.Method.Type
    nucleus_index = int(params.WhichExperiment.Nucleus.Index[0])
    if ('Cascade' == method) and (nucleus_index == 1):
        save_BoundingBox_Hierarchy(params, prediction)

    return True

def loadModel(params):
    model = architecture(params.WhichExperiment.HardParams.Model)
    model.load_weights(params.directories.Train.Model + '/model_weights.h5')       
    return model

def testingExeriment(model, Data, params):
    
    class prediction:
        Test = ''
        Train = ''

    def predictingTestSubject(DataSubj, subject, ResultDir):

        def postProcessing(pred1Class, origMsk1N, NucleiIndex):

            def binarizing(pred1N):
                # Thresh = max( skimage.filters.threshold_otsu(pred1N) ,0.2)  if len(np.unique(pred1N)) != 1 else 0
                return pred1N  > 0.5 # Thresh

            def cascade_paddingToOrigSize(im):
                if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and 1 not in params.WhichExperiment.Nucleus.Index:
                    im = np.pad(im, subject.NewCropInfo.PadSizeBackToOrig, 'constant')
                    # Padding2, crd = paddingNegativeFix(im.shape, Padding2)
                    # im = im[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]
                return im

            def closeMask(mask):
                struc = ndimage.generate_binary_structure(3,2)
                return ndimage.binary_closing(mask, structure=struc)
            
            pred = np.squeeze(pred1Class)
            pred2 = binarizing(pred)
            
            # pred = cascade_paddingToOrigSize(pred)                        
            
            if params.WhichExperiment.HardParams.Model.Method.ImClosePrediction: 
                pred2 = closeMask(pred2)

            pred2 = smallFuncs.extracting_the_biggest_object(pred2)
            
            label_mask = np.transpose( origMsk1N , params.WhichExperiment.Dataset.slicingInfo.slicingOrder)
            Dice = [ NucleiIndex , smallFuncs.mDice(pred2 , binarizing(label_mask)) ]

            # This can be changed to from pred2 to pred, for percision-recall curves
            pred = cascade_paddingToOrigSize(pred2)
            pred = np.transpose(pred , params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse)                                                              

            return pred, Dice

        def savingOutput(pred1N_BtO, NucleiIndex):
            dirSave = smallFuncs.mkDir(ResultDir)
            nucleus = smallFuncs.Nuclei_Class(index=NucleiIndex).name
            smallFuncs.saveImage( pred1N_BtO , DataSubj.Affine, DataSubj.Header, dirSave + '/' + nucleus + '.nii.gz')
            return dirSave, nucleus

        def applyPrediction():

            num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes
            if not params.WhichExperiment.HardParams.Model.Method.havingBackGround_AsExtraDimension:
                num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes + 1

            def loopOver_AllClasses_postProcessing(pred):

                Dice = np.zeros((num_classes-1,2))
                ALL_pred = []
                for cnt in range(num_classes-1):

                    pred1N_BtO, Dice[cnt,:] = postProcessing(pred[...,cnt], DataSubj.OrigMask[...,cnt] , params.WhichExperiment.Nucleus.Index[cnt] )
                    
                    if int(params.WhichExperiment.Nucleus.Index[0]) == 1:
                        ALL_pred = np.concatenate( (ALL_pred , pred1N_BtO[...,np.newaxis]) , axis=3) if cnt > 0 else pred1N_BtO[...,np.newaxis]

                    dirSave, nucleusName = savingOutput(pred1N_BtO, params.WhichExperiment.Nucleus.Index[cnt])
                
                Dir_Dice = dirSave + '/Dice_All.txt' if (params.WhichExperiment.HardParams.Model.MultiClass.Mode and num_classes > 2 ) else dirSave + '/Dice_' + nucleusName + '.txt'
                np.savetxt(Dir_Dice ,Dice,fmt='%1.1f %1.4f')
                return ALL_pred

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
            
            def downsample_Mask(Mask , scale):

                szM = Mask.shape
                
                Mask3  = np.zeros( (szM[0] , int(szM[1]/scale) , int(szM[2]/scale) , szM[3])  )

                newShape = int(szM[1]/scale) , int(szM[2]/scale)

                tform = AffineTransform(scale=(1/scale, 1/scale))
                for i in range(Mask3.shape[0]):
                    for ch in range(Mask3.shape[3]):
                        Mask3[i ,: ,: ,ch]  = warp( np.squeeze(Mask[i ,: ,: ,ch]) ,  tform.inverse, output_shape=newShape, order=0)
                
                return  Mask3

            im = DataSubj.Image.copy()
            predF = model.predict(im)
            # score = model.evaluate(DataSubj.Image, DataSubj.Mask)
            pred = predF[...,:num_classes-1]
            if len(pred.shape) == 3: pred = np.expand_dims(pred,axis=3)

           
            pred = np.transpose(pred,[1,2,0,3])
            if len(pred.shape) == 3: pred = np.expand_dims(pred,axis=3)

            pred = unPadding(pred, subject.Padding)
            return loopOver_AllClasses_postProcessing(pred)

        return applyPrediction()

    def loopOver_Predicting_TestSubjects(DataTest):
        prediction = {}
        # ResultDir = params.directories.Test.Result
        for name in tqdm(DataTest,desc='predicting test subjects'):
            subject = params.directories.Test.Input.Subjects[name]
            ResultDir = subject.address + '/' + params.UserInfo['thalamic_side'].active_side + '/sd' + str(params.WhichExperiment.Dataset.slicingInfo.slicingDim) + '/'
            prediction[name] = predictingTestSubject(DataTest[name], subject , ResultDir)

        return prediction

    def loopOver_Predicting_TestSubjects_Sagittal(DataTest):
        prediction = {}
        # ResultDir = params.directories.Test.Result.replace('/sd2','/sd0')
        for name in tqdm(DataTest,desc='predicting test subjects sagittal'):
            subject = params.directories.Test.Input.Subjects[name]
            ResultDir = subject.address + '/' + params.UserInfo['thalamic_side'].active_side + '/sd0/'
            prediction[name] = predictingTestSubject(DataTest[name], subject , ResultDir)

        return prediction

    def loopOver_Predicting_TrainSubjects(DataTrain):
        prediction = {}
        if params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data or (int(params.WhichExperiment.Nucleus.Index[0]) == 1):
            ResultDir = smallFuncs.mkDir(params.directories.Test.Result + '/TrainData_Output')
            for name in tqdm(DataTrain,desc='predicting train subjects'):
                subject = params.directories.Train.Input.Subjects[name]
                prediction[name] = predictingTestSubject(DataTrain[name], subject , ResultDir + '/' + subject.subjectName + '/')

        return prediction

    def loopOver_Predicting_TrainSubjects_Sagittal(DataTrain):
        prediction = {}
        if (params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data) or (int(params.WhichExperiment.Nucleus.Index[0]) == 1):
            ResultDir = smallFuncs.mkDir(params.directories.Test.Result.replace('/sd2','/sd0') + '/TrainData_Output')
            for name in tqdm(DataTrain,desc='predicting train subjects sagittal'):
                subject = params.directories.Train.Input.Subjects[name]
                prediction[name] = predictingTestSubject(DataTrain[name], subject , ResultDir + '/' + subject.subjectName + '/')

        return prediction

    prediction.Test  = loopOver_Predicting_TestSubjects(Data.Test)
    prediction.Train = loopOver_Predicting_TrainSubjects(Data.Train_ForTest)

    if params.WhichExperiment.Nucleus.Index[0] == 1 and params.WhichExperiment.Dataset.slicingInfo.slicingDim == 2:
        prediction.Sagittal_Test  = loopOver_Predicting_TestSubjects_Sagittal(Data.Sagittal_Test)
        prediction.Sagittal_Train = loopOver_Predicting_TrainSubjects_Sagittal(Data.Sagittal_Train_ForTest)

    return prediction

def trainingExperiment(Data, params):

    def func_CallBacks(params):
        batch_size  = params.WhichExperiment.HardParams.Model.batch_size
        Dir_Save = params.directories.Train.Model
        mode = 'max'
        monitor = 'val_mDice'       
        
        checkpointer = keras.callbacks.ModelCheckpoint(filepath= Dir_Save + '/best_model_weights.h5', \
            monitor = 'val_mDice' , verbose=1, save_best_only=True, mode=mode)

        Reduce_LR = keras.callbacks.ReduceLROnPlateau(monitor = monitor, factor=0.5, min_delta=0.001 , patience=15, verbose=1, \
            save_best_only=True, mode=mode , min_lr=1e-6 , )
        
        def step_decay_schedule(initial_lr=params.WhichExperiment.HardParams.Model.Learning_Rate, decay_factor=0.5, step_size=18):
            def schedule(epoch):
                return initial_lr * (decay_factor ** np.floor(epoch/step_size))

            return keras.callbacks.LearningRateScheduler(schedule, verbose=1)
        
        EarlyStopping = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=40, verbose=1, mode=mode, \
            baseline=0, restore_best_weights=True)

        if params.WhichExperiment.HardParams.Model.LR_Scheduler: return [checkpointer  , EarlyStopping , Reduce_LR]
        else: return [checkpointer  , EarlyStopping] # TQDMCallback()
        
    def saveReport(DirSave, name , data, method):

        def savePickle(Dir, data):
            f = open(Dir,"wb")
            pickle.dump(data,f)
            f.close()

        savePickle(DirSave + '/' + name + '.pkl', data)

    def modelTrain_Unet(Data, params, modelS):
        ModelParam = params.WhichExperiment.HardParams.Model
                    
        def modelInitialize(model):            
        
            FM = '/FM' + str(params.WhichExperiment.HardParams.Model.Layer_Params.FirstLayer_FeatureMap_Num)
            NN = '/' + params.WhichExperiment.Nucleus.name    
            SD = '/sd' + str(params.WhichExperiment.Dataset.slicingInfo.slicingDim) 
            initialization = params.WhichExperiment.HardParams.Model.Initialize
            code_address = smallFuncs.dir_check(params.UserInfo['experiment'].code_address)

            if initialization.init_address:
                init_address = smallFuncs.dir_check(initialization.init_address) + FM + NN + SD + '/model_weights.h5'
            else:
                modDef = initialization.modality_default.lower()
                net_name = 'SRI' if modDef == 'wmn' else 'WMn'

                init_address = code_address + 'Trained_Models/' + net_name + FM + NN + SD + '/model_weights.h5'

            try: 
                model.load_weights(init_address)
                print(' --- initialization succesfull')
            except: 
                print('initialization failed')

            return model  

        # if params.WhichExperiment.HardParams.Model.Initialize:
        model = modelInitialize(modelS)
        
        if len(params.WhichExperiment.HardParams.Machine.GPU_Index) > 1: model = multi_gpu_model(model)

        def saveModel_h5(model, modelS):       
            smallFuncs.mkDir(params.directories.Train.Model)
            if params.WhichExperiment.HardParams.Model.Method.save_Best_Epoch_Model:                
                model.load_weights(params.directories.Train.Model + '/best_model_weights.h5')

            model.save(params.directories.Train.Model + '/orig_model.h5', overwrite=True, include_optimizer=False )
            model.save_weights(params.directories.Train.Model + '/orig_model_weights.h5', overwrite=True )

            modelS.save(params.directories.Train.Model + '/model.h5', overwrite=True, include_optimizer=False )
            modelS.save_weights(params.directories.Train.Model + '/model_weights.h5', overwrite=True )

            return model
                
        def modelFit(model):

                
            class_weights = func_class_weights(params.WhichExperiment.HardParams.Model.Layer_Params.class_weight.Mode, Data.Train.Mask)
            _, loss_tag = LossFunction.LossInfo(params.UserInfo['simulation'].lossFunction_Index)

            if  'My' in loss_tag: model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss(class_weights) , metrics=ModelParam.metrics)
            else: model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
                
            def func_modelParams():
                callbacks = func_CallBacks(params)
                                
                batch_size       = params.WhichExperiment.HardParams.Model.batch_size
                epochs           = params.WhichExperiment.HardParams.Model.epochs 
                valSplit_Per     = params.WhichExperiment.Dataset.Validation.percentage
                verbose          = params.WhichExperiment.HardParams.Model.verbose                
                Validation_fromKeras = params.WhichExperiment.Dataset.Validation.fromKeras
            
                return callbacks , batch_size, epochs, valSplit_Per, verbose , Validation_fromKeras
            callbacks , batch_size, epochs, valSplit_Per, verbose , Validation_fromKeras = func_modelParams()

            def func_without_Generator():
                if params.WhichExperiment.HardParams.Model.Layer_Params.class_weight.Mode and ('My' not in loss_tag):                    
                    hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, class_weight=class_weights , batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Mask), verbose=verbose, callbacks=callbacks) # , callbacks=[TQDMCallback()])                        
                else:                 
                    hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Mask), verbose=verbose, callbacks=callbacks) # , callbacks=[TQDMCallback()])        

                if ModelParam.showHistory: print(hist.history)

                return hist

            hist = func_without_Generator()


            return model, hist                         
        model, hist = modelFit(model)
        model = saveModel_h5(model, modelS)
        return model, hist

    smallFuncs.Saving_UserInfo(params.directories.Train.Model, params)
    model = architecture(params.WhichExperiment.HardParams.Model)
    model, hist = modelTrain_Unet(Data, params, model)
    saveReport(params.directories.Train.Model , 'hist_history' , hist.history , 'pickle')
    return model

def save_BoundingBox_Hierarchy(params, PRED):

    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    def save_BoundingBox(PreStageMask, subject, mode , dirr):

        def checkBordersOnBoundingBox(Sz , BB , gapS):
            return [   [   np.max([BB[d][0]-gapS,0])  ,   np.min( [BB[d][1]+gapS,Sz[d]])   ]  for d in range(3) ]

        imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
        # if 'train' in mode: 
        #     dirr += '/TrainData_Output'

        for ch in range(PreStageMask.shape[3]):
            BB = smallFuncs.findBoundingBox(PreStageMask[...,ch])
            gapDilation = params.WhichExperiment.Dataset.gapDilation
            BBd = [  [BB[ii][0] - gapDilation , BB[ii][1] + gapDilation] for ii in range(len(BB))]
            BBd = checkBordersOnBoundingBox(imF.shape , BBd , 0)            

            BB = checkBordersOnBoundingBox(imF.shape , BB , params.WhichExperiment.Dataset.gapOnSlicingDimention)

            nucleusName = smallFuncs.Nuclei_Class(index=params.WhichExperiment.Nucleus.Index[ch]).name
            np.savetxt(dirr + '/BB_' + nucleusName + '.txt',np.concatenate((BB,BBd),axis=1),fmt='%d')

    nucleus = params.WhichExperiment.Nucleus.name
    def loop_Subjects(PRED, mode):
        if PRED:
            if 'train' in mode:
                Subjects = params.directories.Train.Input.Subjects 
                for name in tqdm(Subjects ,desc='saving BB ' + ' ' + mode + nucleus):
                    save_BoundingBox(PRED[name] , Subjects[name] , mode , params.directories.Test.Result + '/TrainData_Output/' + name + '/')

            elif 'test' in mode:
                Subjects = params.directories.Test.Input.Subjects 
                for name in tqdm(Subjects ,desc='saving BB ' + ' ' + mode + nucleus):
                    ResultDir = Subjects[name].address + '/' + params.UserInfo['thalamic_side'].active_side + '/sd' + str(params.WhichExperiment.Dataset.slicingInfo.slicingDim) + '/'
                    save_BoundingBox(PRED[name] , Subjects[name] , mode , ResultDir)

    def loop_Subjects_Sagittal(PRED, mode):
        if PRED:

            if 'train' in mode:
                Subjects = params.directories.Train.Input.Subjects 
                for name in tqdm(Subjects ,desc='saving BB ' + ' ' + mode + nucleus):
                    save_BoundingBox(PRED[name] , Subjects[name] , mode , params.directories.Test.Result.replace('/sd2','/sd0') + '/TrainData_Output/' + name)

            elif 'test' in mode:
                Subjects = params.directories.Test.Input.Subjects 
                for name in tqdm(Subjects ,desc='saving BB ' + ' ' + mode + nucleus):
                    ResultDir = Subjects[name].address + '/' + params.UserInfo['thalamic_side'].active_side + '/sd0/'
                    save_BoundingBox(PRED[name] , Subjects[name] , mode , ResultDir)



    loop_Subjects(PRED.Test, 'test')
    loop_Subjects(PRED.Train, 'train')

    if params.WhichExperiment.Nucleus.Index[0] == 1 and params.WhichExperiment.Dataset.slicingInfo.slicingDim == 2:
        loop_Subjects_Sagittal(PRED.Sagittal_Test, 'test')
        loop_Subjects_Sagittal(PRED.Sagittal_Train, 'train')        

def architecture(ModelParam):
    input_shape = tuple(ModelParam.InputDimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)

    def Res_Unet(ModelParam):  #  Conv -> BatchNorm -> Relu ) -> (Conv -> BatchNorm -> Relu)  -> maxpooling  -> Dropout
                    
        LP = ModelParam.Layer_Params        
        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num

        padding     = ModelParam.Layer_Params.ConvLayer.padding
        NLayers     = ModelParam.num_Layers
        num_classes = ModelParam.MultiClass.num_classes
        pool_size   = ModelParam.Layer_Params.MaxPooling.pool_size

        def Layer(featureMaps, trainable, input):
            conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(input)
            conv = KLayers.BatchNormalization()(conv)  
            return KLayers.Activation(AC.layers)(conv) 

        def Unet_sublayer_Contracting(inputs):
            def main_USC(WBp, nL):
                trainable = True
                featureMaps = FM*(2**nL)

                conv = Layer(featureMaps, trainable, WBp)
                conv = Layer(featureMaps, trainable, conv)                                              
                
                #! Residual Part
                conv = KLayers.merge.concatenate( [WBp, conv] , axis=3)  

                pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)                                
                
                if trainable: pool = KLayers.Dropout(DT.Value)(pool)  
                                
                return pool, conv
            
            for nL in range(NLayers -1):  
                if nL == 0: WB, Conv_Out = inputs , {}
                WB, Conv_Out[nL+1] = main_USC(WB, nL)  

            return WB, Conv_Out

        def Unet_sublayer_Expanding(WB , Conv_Out):
            def main_USE(WBp, nL, contracting_Info):
                trainable = TF.U_Net4.Expanding[nL] if TF.Mode else True
                featureMaps = FM*(2**nL)

                WBp = KLayers.Conv2DTranspose(featureMaps, kernel_size=KN.convTranspose, strides=(2,2), padding=padding, activation=AC.layers, trainable=trainable)(WBp)
                UP = KLayers.merge.concatenate( [WBp, contracting_Info[nL+1]] , axis=3)

                conv = Layer(featureMaps, trainable, UP)
                conv = Layer(featureMaps, trainable, conv)
                
                #! Residual Part
                conv = KLayers.merge.concatenate( [UP, conv] , axis=3)  

                if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers -1)):  
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(input, nL):
            trainable = TF.U_Net4.Middle if TF.Mode else True
            featureMaps = FM*(2**nL)

            WB = Layer(featureMaps, trainable, input)
            WB = Layer(featureMaps, trainable, WB)

            #! Residual Part
            WB = KLayers.merge.concatenate( [input, WB] , axis=3)  


            if DT.Mode and trainable: WB = KLayers.Dropout(DT.Value)(WB)
            return WB
                
        inputs = KLayers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB , NLayers-1)

        WB = Unet_sublayer_Expanding(WB , Conv_Out)

        final = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])
        
    def Res_Unet2(ModelParam):  #  Conv -> BatchNorm -> Relu ) -> (Conv -> BatchNorm -> Relu)  -> maxpooling  -> Dropout
                    
        LP = ModelParam.Layer_Params        
        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num

        padding     = ModelParam.Layer_Params.ConvLayer.padding
        NLayers     = ModelParam.num_Layers
        num_classes = ModelParam.MultiClass.num_classes
        pool_size   = ModelParam.Layer_Params.MaxPooling.pool_size

        def Layer(featureMaps, trainable, input):
            conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(input)
            conv = KLayers.BatchNormalization()(conv)  
            return KLayers.Activation(AC.layers)(conv) 

        def Unet_sublayer_Contracting(inputs):
            def main_USC(WBp, nL):
                trainable = True
                featureMaps = FM*(2**nL)

                conv = Layer(featureMaps, trainable, WBp)
                conv = Layer(featureMaps, trainable, conv)                                              
                
                #! Residual Part
                if nL > 0: conv = KLayers.merge.concatenate( [WBp, conv] , axis=3)  

                pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)                                
                
                if trainable: pool = KLayers.Dropout(DT.Value)(pool)  
                                
                return pool, conv
            
            for nL in range(NLayers -1):  
                if nL == 0: WB, Conv_Out = inputs , {}
                WB, Conv_Out[nL+1] = main_USC(WB, nL)  

            return WB, Conv_Out

        def Unet_sublayer_Expanding(WB , Conv_Out):
            def main_USE(WBp, nL, contracting_Info):
                trainable = True
                featureMaps = FM*(2**nL)

                WBp = KLayers.Conv2DTranspose(featureMaps, kernel_size=KN.convTranspose, strides=(2,2), padding=padding, activation=AC.layers, trainable=trainable)(WBp)
                UP = KLayers.merge.concatenate( [WBp, contracting_Info[nL+1]] , axis=3)

                conv = Layer(featureMaps, trainable, UP)
                conv = Layer(featureMaps, trainable, conv)
                
                #! Residual Part
                conv = KLayers.merge.concatenate( [UP, conv] , axis=3)  

                if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers -1)):  
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(input, nL):
            trainable = True
            featureMaps = FM*(2**nL)

            WB = Layer(featureMaps, trainable, input)
            WB = Layer(featureMaps, trainable, WB)

            #! Residual Part
            WB = KLayers.merge.concatenate( [input, WB] , axis=3)  


            if DT.Mode and trainable: WB = KLayers.Dropout(DT.Value)(WB)
            return WB
                
        inputs = KLayers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB , NLayers-1)

        WB = Unet_sublayer_Expanding(WB , Conv_Out)

        final = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    if  ModelParam.architectureType == 'Res_Unet':
        model = Res_Unet(ModelParam)
    elif  ModelParam.architectureType == 'Res_Unet2':
        model = Res_Unet2(ModelParam)        

    model.summary()

    return model

def func_classWeights(params , Data):

    if params.WhichExperiment.HardParams.Model.Layer_Params.class_weight.Mode:
        Sz = Data.Train.Mask.shape
        a = np.sum(Data.Train.Mask[...,0] < 0.5) / (Sz[0]*Sz[1]*Sz[2])
        return {0:1-a , 1:a}
    else: return {0:1 , 1:1}
   