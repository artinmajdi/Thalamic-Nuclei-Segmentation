import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras import models as kerasmodels
# from keras import layers
# from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback # , TQDMNotebookCallback
import numpy as np
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import preprocess.croppingA as croppingA
import modelFuncs.Metrics as Metrics
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
from keras.utils import multi_gpu_model, multi_gpu_utils
import h5py
import keras.layers as KLayers
from sklearn.utils import class_weight
# import json
from keras.preprocessing.image import ImageDataGenerator
import modelFuncs.LossFunction as LossFunction
from skimage.transform import AffineTransform , warp

def check_Run(params, Data):

    # os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    model      = trainingExperiment(Data, params) if not params.preprocess.TestOnly else loadModel(params)
    prediction = testingExeriment(model, Data, params)

    if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and (int(params.WhichExperiment.Nucleus.Index[0]) == 1):
        save_BoundingBox_Hierarchy(params, prediction)

    return True

def loadModel(params):
    model = architecture(params.WhichExperiment.HardParams.Model)
    model.load_weights(params.directories.Train.Model + '/model_weights.h5')
    
    # model = kerasmodels.load_model(params.directories.Train.Model + '/model.h5')
    ModelParam = params.WhichExperiment.HardParams.Model
    model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
    return model

def testingExeriment(model, Data, params):
       
    def func_dice_name():
        if 'HCascade' in params.WhichExperiment.HardParams.Model.Method.Type: 
                                    
            if 1.2 in params.WhichExperiment.Nucleus.Index:
                dice_tag = 'All_Groups'                
            else:
                BB = smallFuncs.Nuclei_Class(1,'HCascade')
                parent = BB.HCascade_Parents_Identifier(params.WhichExperiment.Nucleus.Index)
                dice_tag = 'All_' + smallFuncs.Nuclei_Class(parent[0],'HCascade').name.replace('_ImClosed','')
                
        else: 
            dice_tag = 'All'  

        return dice_tag
    dice_tag = func_dice_name()

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

            def closeMask(mask):
                struc = ndimage.generate_binary_structure(3,2)
                return ndimage.binary_closing(mask, structure=struc)

            pred1N = binarizing( np.squeeze(pred1Class) )
            # pred1N = np.squeeze(pred1Class) 
            
            if params.WhichExperiment.HardParams.Model.Method.ImClosePrediction: 
                pred1N = closeMask(pred1N)
            
            pred1N_origShape = cascade_paddingToOrigSize(pred1N)
            pred1N_origShape = np.transpose(pred1N_origShape,params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse)

            Dice = [ NucleiIndex , smallFuncs.mDice(pred1N_origShape , binarizing(origMsk1N)) ]

            return pred1N_origShape, Dice

        def savingOutput(pred1N_BtO, NucleiIndex):
            dirSave = smallFuncs.mkDir(ResultDir + '/' + subject.subjectName)
            nucleusName, _ , _ = smallFuncs.NucleiSelection(NucleiIndex)

            smallFuncs.saveImage( pred1N_BtO , DataSubj.Affine, DataSubj.Header, dirSave + '/' + nucleusName + '.nii.gz')
            # print(dirSave)
            return dirSave, nucleusName

        def applyPrediction():

            num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes
            if not params.WhichExperiment.HardParams.Model.Method.havingBackGround_AsExtraDimension:
                num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes + 1

            def loopOver_AllClasses_postProcessing(pred):

                # TODO "pred1N_BtO" needs to concatenate for different classes
                Dice = np.zeros((num_classes-1,2))
                for cnt in range(num_classes-1):

                    # ! where I added the concatenation for different masks
                    pred1N_BtO, Dice[cnt,:] = postProcessing(pred[...,cnt], DataSubj.OrigMask[...,cnt] , params.WhichExperiment.Nucleus.Index[cnt] )
                    
                    ALL_pred = np.concatenate( (ALL_pred , pred1N_BtO[...,np.newaxis]) , axis=3) if cnt > 0 else pred1N_BtO[...,np.newaxis]

                    dirSave, nucleusName = savingOutput(pred1N_BtO, params.WhichExperiment.Nucleus.Index[cnt])
                
                Dir_Dice = dirSave + '/Dice_' + dice_tag + '.txt' if params.WhichExperiment.HardParams.Model.MultiClass.Mode else dirSave + '/Dice_' + nucleusName + '.txt'
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


            if params.WhichExperiment.HardParams.Model.Upsample.Mode:
                scale = params.WhichExperiment.HardParams.Model.Upsample.Scale
                pred = downsample_Mask(pred , scale)
           
            pred = np.transpose(pred,[1,2,0,3])
            if len(pred.shape) == 3: pred = np.expand_dims(pred,axis=3)

            pred = unPadding(pred, subject.Padding)
            return loopOver_AllClasses_postProcessing(pred)

        return applyPrediction()

    def loopOver_Predicting_TestSubjects(DataTest):
        prediction = {}
        ResultDir = params.directories.Test.Result
        for name in tqdm(DataTest,desc='predicting test subjects'):
            prediction[name] = predictingTestSubject(DataTest[name], params.directories.Test.Input.Subjects[name] , ResultDir)

        return prediction

    def loopOver_Predicting_TestSubjects_Sagittal(DataTest):
        prediction = {}
        ResultDir = params.directories.Test.Result.replace('/sd2','/sd0')
        for name in tqdm(DataTest,desc='predicting test subjects sagittal'):
            prediction[name] = predictingTestSubject(DataTest[name], params.directories.Test.Input_Sagittal.Subjects[name] , ResultDir)

        return prediction

    def loopOver_Predicting_TrainSubjects(DataTrain):
        prediction = {}
        if (params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data) or ( 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and (int(params.WhichExperiment.Nucleus.Index[0]) == 1)):
            ResultDir = smallFuncs.mkDir(params.directories.Test.Result + '/TrainData_Output')
            for name in tqdm(DataTrain,desc='predicting train subjects'):
                prediction[name] = predictingTestSubject(DataTrain[name], params.directories.Train.Input.Subjects[name] , ResultDir)

        return prediction

    def loopOver_Predicting_TrainSubjects_Sagittal(DataTrain):
        prediction = {}
        if (params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data) or ( 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and (int(params.WhichExperiment.Nucleus.Index[0]) == 1)):
            ResultDir = smallFuncs.mkDir(params.directories.Test.Result.replace('/sd2','/sd0') + '/TrainData_Output')
            for name in tqdm(DataTrain,desc='predicting train subjects sagittal'):
                prediction[name] = predictingTestSubject(DataTrain[name], params.directories.Train.Input_Sagittal.Subjects[name] , ResultDir)

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
        
        A = params.directories.Train.model_Tag if params.WhichExperiment.HardParams.Model.Transfer_Learning.Mode else ''
        checkpointer = keras.callbacks.ModelCheckpoint(filepath= Dir_Save + '/best_model_weights' + A + '.h5', \
            monitor = 'val_mDice' , verbose=1, save_best_only=True, mode=mode)

        # Reduce_LR = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5, min_delta=0.005 , patience=4, verbose=1, \
        #     save_best_only=True, mode='min' , min_lr=0.9e-4 , )
        
        def step_decay_schedule(initial_lr=1e-2, decay_factor=0.5, step_size=70):
            def schedule(epoch):
                return initial_lr * (decay_factor ** np.floor(epoch/step_size))            
            return keras.callbacks.LearningRateScheduler(schedule)
        
        Reduce_LR = step_decay_schedule()

        # Progbar = keras.callbacks.Progba
        EarlyStopping = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=20, verbose=1, mode=mode, \
            baseline=0, restore_best_weights=True)

        # TensorBoard = keras.callbacks.TensorBoard(log_dir= Dir_Save + '/logs', histogram_freq=1, batch_size=batch_size, \
        #     write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, \
                # embeddings_data=None, update_freq='epoch')

        
        if params.UserInfo['simulation'].LR_Scheduler: return [checkpointer  , EarlyStopping , Reduce_LR]
        else: return [checkpointer  , EarlyStopping] # TQDMCallback()
        
    def saveReport(DirSave, name , data, method):

        def savePickle(Dir, data):
            f = open(Dir,"wb")
            pickle.dump(data,f)
            f.close()

        if 'pickle' in method: savePickle(DirSave + '/' + name + '.pkl', data)
        elif 'mat'  in method: mat4py.savemat(DirSave + '/' + name + '.mat', data)
        elif 'csv'  in method: pd.DataFrame(data=data,columns=list(data.keys())).to_csv( DirSave + '/' + name + '.csv')

    def saveTrainInfo(hist,a, params):

        # if hist:
        # hist.params['trainingTime']     = time() - a
        # hist.params['InputDimensionsX'] = params.WhichExperiment.HardParams.Model.InputDimensions[0]
        # hist.params['InputDimensionsY'] = params.WhichExperiment.HardParams.Model.InputDimensions[1]
        # hist.params['InputDimensionsZ'] = params.WhichExperiment.HardParams.Model.InputDimensions[2]
        # hist.params['num_Layers']       = params.WhichExperiment.HardParams.Model.num_Layers

        saveReport(params.directories.Train.Model , 'hist_history' , hist.history , params.UserInfo['SaveReportMethod'])
        # saveReport(params.directories.Train.Model , 'hist_model'   , hist.model   , params.UserInfo['SaveReportMethod'])
        # saveReport(params.directories.Train.Model , 'hist_params'  , hist.params  , params.UserInfo['SaveReportMethod'])

    def modelTrain_Unet(Data, params, modelS):
        ModelParam = params.WhichExperiment.HardParams.Model
                    
        def modelInitialize(model):            
            TP = params.directories.Train
            if params.WhichExperiment.HardParams.Model.Transfer_Learning.Mode:
                model.load_weights(TP.Model + '/model_weights.h5')
                print(' --- initialized from older Model <- -> Transfer Learning ')                                
            else:
                
                Initialize = params.WhichExperiment.HardParams.Model.Initialize
                                        
                if Initialize.FromOlderModel and os.path.exists(TP.Model + '/model_weights.h5'):
                    try: 
                        model.load_weights(TP.Model + '/model_weights.h5')
                        print(' --- initialized from older Model  ')
                    except: print('initialized from older Model failed')

                elif Initialize.FromThalamus and params.WhichExperiment.Nucleus.Index[0] != 1 and os.path.exists(TP.Model_Thalamus + '/model_weights.h5'):
                    try: 
                        model.load_weights(TP.Model_Thalamus + '/model_weights.h5')
                        print(' --- initialized from Thalamus  ')
                    except: print('initialized from Thalamus failed')
                        

                elif Initialize.From_3T and os.path.exists(TP.Model_3T + '/model_weights.h5'):
                    print('-----' , TP.Model_3T + '/model_weights.h5')
                    try:
                        model.load_weights(TP.Model_3T + '/model_weights.h5')
                        print(' --- initialized from Model_3T' , TP.Model_3T)
                    except: print('initialized from 3T failed')

                elif Initialize.From_7T and os.path.exists(TP.Model_7T + '/model_weights.h5'):
                    try:
                        model.load_weights(TP.Model_7T + '/model_weights.h5')
                        print(' --- initialized from Model_7T' , TP.Model_7T)
                    except: print('initialized from From_7T failed')  

                elif Initialize.From_CSFn and os.path.exists(TP.Model_CSFn + '/model_weights.h5'):
                    try:
                        model.load_weights(TP.Model_CSFn + '/model_weights.h5')
                        print(' --- initialized from Model_CSFn' , TP.Model_CSFn)
                    except: print('initialized from From_CSFn failed') 


            print('------------------------------------------------------------------')
            return model  

        model = modelInitialize(modelS)
        
        if len(params.WhichExperiment.HardParams.Machine.GPU_Index) > 1: model = multi_gpu_model(model)

        def saveModel_h5(model, modelS):       
            smallFuncs.mkDir(params.directories.Train.Model)
            
            A = params.directories.Train.model_Tag if params.WhichExperiment.HardParams.Model.Transfer_Learning.Mode else ''

            if params.WhichExperiment.HardParams.Model.Method.save_Best_Epoch_Model:                
                model.load_weights(params.directories.Train.Model + '/best_model_weights' + A + '.h5')

            # model.save(params.directories.Train.Model + '/model' + A + '.h5', overwrite=True, include_optimizer=False )
            # model.save_weights(params.directories.Train.Model + '/model_weights' + A + '.h5', overwrite=True )

            modelS.save(params.directories.Train.Model + '/model' + A + '.h5', overwrite=True, include_optimizer=False )
            modelS.save_weights(params.directories.Train.Model + '/model_weights' + A + '.h5', overwrite=True )

            keras.utils.plot_model(modelS,to_file=params.directories.Train.Model+'/Architecture.png',show_layer_names=True,show_shapes=True)

            return model
                
        def modelFit(model):

            def func_class_weights():
                sz = Data.Train.Mask.shape
                NUM_CLASSES = sz[3]
                class_weights = np.ones(NUM_CLASSES)

                #! equal weights
                if params.WhichExperiment.HardParams.Model.Layer_Params.class_weight.Mode:
                    for ix in range(NUM_CLASSES):                                        
                        TRUE_Count = len(np.where(Data.Train.Mask[...,ix] > 0.5)[0])
                        NUM_SAMPLES = np.prod(sz[:3])
                        class_weights[ix] = NUM_SAMPLES / (NUM_CLASSES*TRUE_Count)

                print('class_weights' , class_weights)
                # # ! zero weight for foreground
                # for i in [0,2,5,7]: class_weights[i] = 0
                # # class_weights = class_weight.compute_class_weight('balanced',classes,y_train)
                return class_weights
                
            class_weights = func_class_weights()
            _, loss_tag = LossFunction.LossInfo(params.UserInfo['lossFunction_Index'])

            if  'My' in loss_tag: model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss(class_weights) , metrics=ModelParam.metrics)
            else: model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
                
            def func_modelParams():
                callbacks = func_CallBacks(params)
                                
                batch_size       = params.WhichExperiment.HardParams.Model.batch_size
                epochs           = params.WhichExperiment.HardParams.Model.epochs 
                valSplit_Per     = params.WhichExperiment.Dataset.Validation.percentage
                verbose          = params.WhichExperiment.HardParams.Model.verbose                
                Validation_fromKeras = params.WhichExperiment.Dataset.Validation.fromKeras
            
                # A = params.directories.Train.model_Tag if params.WhichExperiment.HardParams.Model.Transfer_Learning.Mode else ''
                # model2.save_weights(params.directories.Train.Model + '/model_weights' + A + '.h5', overwrite=True )
                return callbacks , batch_size, epochs, valSplit_Per, verbose , Validation_fromKeras
            callbacks , batch_size, epochs, valSplit_Per, verbose , Validation_fromKeras = func_modelParams()
                          
            def func_RunGenerator():
                
                info = {'mode': 'train',  'params': params}
                trainGenerator = DataGenerator_Class(**info)
                
                # info = {'mode': 'validation',  'params': params}
                # valGenerator = DataGenerator_Class(**info)
                                    
                hist = model.fit_generator(trainGenerator, epochs=epochs, verbose=verbose, callbacks=callbacks,\
                     validation_data=(Data.Validation.Image, Data.Validation.Mask), workers=10, use_multiprocessing=False, shuffle=True, initial_epoch=0) # 

                return hist

            def func_without_Generator():
                # keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "engr-bilgin01s:6064"))   
                if params.WhichExperiment.HardParams.Model.Layer_Params.class_weight.Mode and ('My' not in loss_tag):                    
                    # if Validation_fromKeras: hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, class_weight=class_weights , batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=valSplit_Per                                , verbose=verbose, callbacks=callbacks) # , callbacks=[TQDMCallback()])                   
                    hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, class_weight=class_weights , batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Mask), verbose=verbose, callbacks=callbacks) # , callbacks=[TQDMCallback()])                        
                else:                 
                    hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Mask), verbose=verbose, callbacks=callbacks) # , callbacks=[TQDMCallback()])        

                if ModelParam.showHistory: print(hist.history)

                return hist

            if params.WhichExperiment.HardParams.Model.DataGenerator.Mode: hist = func_RunGenerator()
            else: hist = func_without_Generator()


            return model, hist                         
        model, hist = modelFit(model)

        model = saveModel_h5(model, modelS)
        # model2 = architecture(params.WhichExperiment.HardParams.Model)

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

    a = time()
    smallFuncs.Saving_UserInfo(params.directories.Train.Model, params)
    model = architecture(params.WhichExperiment.HardParams.Model)

    # if 'U-Net' in params.WhichExperiment.HardParams.Model.architectureType:
    model, hist = modelTrain_Unet(Data, params, model)
    saveReport(params.directories.Train.Model , 'hist_history' , hist.history , 'pickle')

    # elif 'FCN_Cropping':
    #     model, hist = modelTrain_Unet(Data, params, model)

    return model

def save_BoundingBox_Hierarchy(params, PRED):

    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    def save_BoundingBox(PreStageMask, subject, mode , dirr):

        # def cropBoundingBoxes(PreStageMask, dirr):

        def checkBordersOnBoundingBox(Sz , BB , gapS):
            return [   [   np.max([BB[d][0]-gapS,0])  ,   np.min( [BB[d][1]+gapS,Sz[d]])   ]  for d in range(3) ]

        # Thalamus_Mask_Dilated = dilateMask( PreStageMask, params.WhichExperiment.Dataset.gapDilation )
        imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
        if 'train' in mode: dirr += '/TrainData_Output'

        for ch in range(PreStageMask.shape[3]):
            BB = smallFuncs.findBoundingBox(PreStageMask[...,ch])
            BB = checkBordersOnBoundingBox(imF.shape , BB , params.WhichExperiment.Dataset.gapOnSlicingDimention)
            gapDilation = params.WhichExperiment.Dataset.gapDilation
            BBd = [  [BB[ii][0] - gapDilation , BB[ii][1] + gapDilation] for ii in range(len(BB))]
            BBd = checkBordersOnBoundingBox(imF.shape , BBd , 0)            

            nucleusName = smallFuncs.Nuclei_Class(index=params.WhichExperiment.Nucleus.Index[ch]).name
            np.savetxt(dirr + '/' + subject.subjectName + '/BB_' + nucleusName + '.txt',np.concatenate((BB,BBd),axis=1),fmt='%d')

            # cropBoundingBoxes(PreStageMask, dirr)

    nucleus = params.WhichExperiment.Nucleus.name
    def loop_Subjects(PRED, mode):
        if PRED:
            Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects        
            for sj in tqdm(Subjects ,desc='saving BB ' + ' ' + mode + nucleus):
                save_BoundingBox(PRED[sj] , Subjects[sj] , mode , params.directories.Test.Result)

    def loop_Subjects_Sagittal(PRED, mode):
        if PRED:
            Subjects = params.directories.Train.Input_Sagittal.Subjects if 'train' in mode else params.directories.Test.Input_Sagittal.Subjects
            for sj in tqdm(Subjects ,desc='saving BB ' + ' ' + mode + nucleus + ' Sagittal'):
                save_BoundingBox(PRED[sj] , Subjects[sj] , mode , params.directories.Test.Result.replace('/sd2','/sd0'))

    loop_Subjects(PRED.Test, 'test')
    loop_Subjects(PRED.Train, 'train')

    if params.WhichExperiment.Nucleus.Index[0] == 1 and params.WhichExperiment.Dataset.slicingInfo.slicingDim == 2:
        loop_Subjects_Sagittal(PRED.Sagittal_Test, 'test')
        loop_Subjects_Sagittal(PRED.Sagittal_Train, 'train')        


def architecture(ModelParam):

    if ModelParam.Upsample.Mode:
        scale , szI = ModelParam.Upsample.Scale   ,  ModelParam.InputDimensions        
        InDim_new = (scale*szI[0] , scale*szI[1] , szI[2] , 1)
        input_shape = tuple(InDim_new[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
    else:
        input_shape = tuple(ModelParam.InputDimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)

    # ModelParam = params.WhichExperiment.HardParams.Model
    def UNet(ModelParam):  # BatchNorm -> (Conv ->  Relu ) -> (Conv -> Relu)  -> maxpooling  -> Dropout
        
        TF = ModelParam.Transfer_Learning
        LP = ModelParam.Layer_Params        
        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num

        # input_shape = tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
        padding     = ModelParam.Layer_Params.ConvLayer.padding
        NLayers     = ModelParam.num_Layers
        num_classes = ModelParam.MultiClass.num_classes
        pool_size   = ModelParam.Layer_Params.MaxPooling.pool_size

        def Unet_sublayer_Contracting(inputs):
            def main_USC(WBp, nL):
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                if LP.batchNormalization:  WBp = KLayers.BatchNormalization()(WBp)
                conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, activation=AC.layers, trainable=trainable)(WBp)
                conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, activation=AC.layers, trainable=trainable)(conv)

                if 1: # np.min(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) > 30: 
                    pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)  
                    if DT.Mode and trainable: pool = KLayers.Dropout(DT.Value)(pool)  
                else: 
                    if DT.Mode and trainable: pool = KLayers.Dropout(DT.Value)(conv)  
                
                              
                # if LP.batchNormalization:  pool = KLayers.BatchNormalization()(pool)
                
                return pool, conv
            
            for nL in range(NLayers -1):  
                if nL == 0: WB, Conv_Out = inputs , {}
                WB, Conv_Out[nL+1] = main_USC(WB, nL)  

            return WB, Conv_Out

        def Unet_sublayer_Expanding(WB , Conv_Out):
            def main_USE(WBp, nL, contracting_Info):
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                if LP.batchNormalization:  WBp = KLayers.BatchNormalization()(WBp)

                # if 1: #  np.min(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) > 30: 
                WBp = KLayers.Conv2DTranspose(featureMaps, kernel_size=KN.convTranspose, strides=(2,2), padding=padding, activation=AC.layers, trainable=trainable)(WBp)
                UP = KLayers.merge.concatenate( [WBp, contracting_Info[nL+1]] , axis=3)
                conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, activation=AC.layers, trainable=trainable)(UP)
                conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, activation=AC.layers, trainable=trainable)(conv)
                
                # if LP.batchNormalization:  conv = KLayers.BatchNormalization()(conv)                                    
                if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers -1)):  
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(WB, nL):
            trainable = False if TF.Mode and nL in TF.FrozenLayers else True
            featureMaps = FM*(2**nL)

            # if LP.batchNormalization: WB = KLayers.BatchNormalization()(WB)  
            WB = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, activation=AC.layers, trainable=trainable)(WB)
            WB = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, activation=AC.layers, trainable=trainable)(WB)
            
            # if LP.batchNormalization: WB = KLayers.BatchNormalization()(WB)            
            if DT.Mode and trainable: WB = KLayers.Dropout(DT.Value)(WB)
            return WB
                
        inputs = KLayers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB , NLayers-1)

        WB = Unet_sublayer_Expanding(WB , Conv_Out)

        final = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    """
    def UNet2(ModelParam):  #  Conv -> BatchNorm -> Relu ) -> (Conv -> BatchNorm -> Relu)  -> Dropout -> maxpooling
                    
        TF = ModelParam.Transfer_Learning
        LP = ModelParam.Layer_Params        
        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num


        # input_shape = tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
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
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                conv = Layer(featureMaps, trainable, WBp)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(WBp)
                # conv = KLayers.BatchNormalization()(conv)  
                # conv = KLayers.Activation(AC.layers)(conv) 

                conv = Layer(featureMaps, trainable, conv)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(conv)
                # conv = KLayers.BatchNormalization()(conv)  
                # conv = KLayers.Activation(AC.layers)(conv) 
                                              
                if trainable: conv = KLayers.Dropout(DT.Value)(conv)  
                pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)                                
                                
                return pool, conv
            
            for nL in range(NLayers -1):  
                if nL == 0: WB, Conv_Out = inputs , {}
                WB, Conv_Out[nL+1] = main_USC(WB, nL)  

            return WB, Conv_Out

        def Unet_sublayer_Expanding(WB , Conv_Out):
            def main_USE(WBp, nL, contracting_Info):
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                WBp = KLayers.Conv2DTranspose(featureMaps, kernel_size=KN.convTranspose, strides=(2,2), padding=padding, activation=AC.layers, trainable=trainable)(WBp)

                UP = KLayers.merge.concatenate( [WBp, contracting_Info[nL+1]] , axis=3)

                conv = Layer(featureMaps, trainable, UP)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(UP)
                # conv = KLayers.BatchNormalization()(conv) 
                # conv = KLayers.Activation(AC.layers)(conv)

                conv = Layer(featureMaps, trainable, conv)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(conv)
                # conv = KLayers.BatchNormalization()(conv) 
                # conv = KLayers.Activation(AC.layers)(conv)
                
                if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers -1)):  
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(WB, nL):
            trainable = False if TF.Mode and nL in TF.FrozenLayers else True
            featureMaps = FM*(2**nL)

            WB = Layer(featureMaps, trainable, WB)
            # WB = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(WB)
            # WB = KLayers.BatchNormalization()(WB) 
            # WB = KLayers.Activation(AC.layers)(WB)

            WB = Layer(featureMaps, trainable, WB)
            # WB = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(WB)
            # WB = KLayers.BatchNormalization()(WB) 
            # WB = KLayers.Activation(AC.layers)(WB)            

            if DT.Mode and trainable: WB = KLayers.Dropout(DT.Value)(WB)
            return WB
                
        inputs = KLayers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB , NLayers-1)

        WB = Unet_sublayer_Expanding(WB , Conv_Out)

        final = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    def UNet3(ModelParam):  #  Conv -> BatchNorm -> Relu -> Dropout ) -> (Conv -> BatchNorm -> Relu -> Dropout) -> maxpooling
            
        TF = ModelParam.Transfer_Learning
        LP = ModelParam.Layer_Params        
        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num

        # input_shape = tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
        padding     = ModelParam.Layer_Params.ConvLayer.padding
        NLayers     = ModelParam.num_Layers
        num_classes = ModelParam.MultiClass.num_classes
        pool_size   = ModelParam.Layer_Params.MaxPooling.pool_size

        def Layer(featureMaps, trainable, input):
            conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(input)
            conv = KLayers.BatchNormalization()(conv)  
            conv = KLayers.Activation(AC.layers)(conv) 
            if trainable: conv = KLayers.Dropout(DT.Value)(conv)  
            return conv

        def Unet_sublayer_Contracting(inputs):
            def main_USC(WBp, nL):
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                conv = Layer(featureMaps, trainable, WBp)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(WBp)
                # conv = KLayers.BatchNormalization()(conv)  
                # conv = KLayers.Activation(AC.layers)(conv) 

                conv = Layer(featureMaps, trainable, conv)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(conv)
                # conv = KLayers.BatchNormalization()(conv)  
                # conv = KLayers.Activation(AC.layers)(conv) 
                                              
                # if trainable: conv = KLayers.Dropout(DT.Value)(conv)  
                pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)                                
                                
                return pool, conv
            
            for nL in range(NLayers -1):  
                if nL == 0: WB, Conv_Out = inputs , {}
                WB, Conv_Out[nL+1] = main_USC(WB, nL)  

            return WB, Conv_Out

        def Unet_sublayer_Expanding(WB , Conv_Out):
            def main_USE(WBp, nL, contracting_Info):
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                WBp = KLayers.Conv2DTranspose(featureMaps, kernel_size=KN.convTranspose, strides=(2,2), padding=padding, activation=AC.layers, trainable=trainable)(WBp)

                UP = KLayers.merge.concatenate( [WBp, contracting_Info[nL+1]] , axis=3)

                conv = Layer(featureMaps, trainable, UP)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(UP)
                # conv = KLayers.BatchNormalization()(conv) 
                # conv = KLayers.Activation(AC.layers)(conv)

                conv = Layer(featureMaps, trainable, conv)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(conv)
                # conv = KLayers.BatchNormalization()(conv) 
                # conv = KLayers.Activation(AC.layers)(conv)
                
                # if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers -1)):  
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(WB, nL):
            trainable = False if TF.Mode and nL in TF.FrozenLayers else True
            featureMaps = FM*(2**nL)

            WB = Layer(featureMaps, trainable, WB)
            # WB = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(WB)
            # WB = KLayers.BatchNormalization()(WB) 
            # WB = KLayers.Activation(AC.layers)(WB)

            WB = Layer(featureMaps, trainable, WB)
            # WB = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(WB)
            # WB = KLayers.BatchNormalization()(WB) 
            # WB = KLayers.Activation(AC.layers)(WB)            

            # if DT.Mode and trainable: WB = KLayers.Dropout(DT.Value)(WB)
            return WB
                
        inputs = KLayers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB , NLayers-1)

        WB = Unet_sublayer_Expanding(WB , Conv_Out)

        final = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    """
    def UNet4(ModelParam):  #  Conv -> BatchNorm -> Relu ) -> (Conv -> BatchNorm -> Relu)  -> maxpooling  -> Dropout
                    
        TF = ModelParam.Transfer_Learning
        LP = ModelParam.Layer_Params        
        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num


        # input_shape = tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
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
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                conv = Layer(featureMaps, trainable, WBp)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(WBp)
                # conv = KLayers.BatchNormalization()(conv)  
                # conv = KLayers.Activation(AC.layers)(conv) 

                conv = Layer(featureMaps, trainable, conv)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(conv)
                # conv = KLayers.BatchNormalization()(conv)  
                # conv = KLayers.Activation(AC.layers)(conv) 
                                              
                
                pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)                                
                
                if trainable: pool = KLayers.Dropout(DT.Value)(pool)  
                                
                return pool, conv
            
            for nL in range(NLayers -1):  
                if nL == 0: WB, Conv_Out = inputs , {}
                WB, Conv_Out[nL+1] = main_USC(WB, nL)  

            return WB, Conv_Out

        def Unet_sublayer_Expanding(WB , Conv_Out):
            def main_USE(WBp, nL, contracting_Info):
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                WBp = KLayers.Conv2DTranspose(featureMaps, kernel_size=KN.convTranspose, strides=(2,2), padding=padding, activation=AC.layers, trainable=trainable)(WBp)
                UP = KLayers.merge.concatenate( [WBp, contracting_Info[nL+1]] , axis=3)

                conv = Layer(featureMaps, trainable, UP)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(UP)
                # conv = KLayers.BatchNormalization()(conv) 
                # conv = KLayers.Activation(AC.layers)(conv)

                conv = Layer(featureMaps, trainable, conv)
                # conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(conv)
                # conv = KLayers.BatchNormalization()(conv) 
                # conv = KLayers.Activation(AC.layers)(conv)
                
                if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers -1)):  
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(WB, nL):
            trainable = False if TF.Mode and nL in TF.FrozenLayers else True
            featureMaps = FM*(2**nL)

            WB = Layer(featureMaps, trainable, WB)
            # WB = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(WB)
            # WB = KLayers.BatchNormalization()(WB) 
            # WB = KLayers.Activation(AC.layers)(WB)

            WB = Layer(featureMaps, trainable, WB)
            # WB = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(WB)
            # WB = KLayers.BatchNormalization()(WB) 
            # WB = KLayers.Activation(AC.layers)(WB)            

            if DT.Mode and trainable: WB = KLayers.Dropout(DT.Value)(WB)
            return WB
                
        inputs = KLayers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB , NLayers-1)

        WB = Unet_sublayer_Expanding(WB , Conv_Out)

        final = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    def FCN(ModelParam):  #  Conv -> BatchNorm -> Relu ) -> (Conv -> BatchNorm -> Relu)  -> maxpooling  -> Dropout
                    
        TF = ModelParam.Transfer_Learning
        LP = ModelParam.Layer_Params        
        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num


        # input_shape = tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
        padding     = ModelParam.Layer_Params.ConvLayer.padding
        NLayers     = ModelParam.num_Layers
        num_classes = ModelParam.MultiClass.num_classes
        pool_size   = ModelParam.Layer_Params.MaxPooling.pool_size

        def Layer(featureMaps, trainable, input):
            conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(input)
            conv = KLayers.BatchNormalization()(conv)  
            return KLayers.Activation(AC.layers)(conv) 

        def CNN_Part(inputs):
            def main_USC(conv, nL):
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                # featureMaps = FM*(2**nL)

                conv = Layer(FM, trainable, conv)
                conv = Layer(FM, trainable, conv)                                                                          
                if trainable: conv = KLayers.Dropout(DT.Value)(conv)                                  
                return conv
            
            for nL in range(NLayers -1):  
                if nL == 0: 
                    conv = main_USC(inputs, nL)
                else: 
                    conv = main_USC(conv, nL)

            return conv
               
        inputs = KLayers.Input(input_shape)
        conv   = CNN_Part(inputs)
        final  = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(conv)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    def FCN_with_SkipConnection(ModelParam):  #  Conv -> BatchNorm -> Relu ) -> (Conv -> BatchNorm -> Relu)  -> maxpooling  -> Dropout
                    
        TF = ModelParam.Transfer_Learning
        LP = ModelParam.Layer_Params        
        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num


        # input_shape = tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
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
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                conv = Layer(featureMaps, trainable, WBp)
                conv = Layer(featureMaps, trainable, conv)                                                              
                # pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)                                
                
                if trainable: conv = KLayers.Dropout(DT.Value)(conv)  
                                
                return conv
            
            for nL in range(NLayers -1):  
                if nL == 0: Conv_Out = {0:inputs}
                Conv_Out[nL+1] = main_USC(Conv_Out[nL], nL)  

            return Conv_Out[nL+1], Conv_Out

        def Unet_sublayer_Expanding(WB , Conv_Out):
            def main_USE(WBp, nL, contracting_Info):
                trainable = False if TF.Mode and nL in TF.FrozenLayers else True
                featureMaps = FM*(2**nL)

                # WBp = KLayers.Conv2DTranspose(featureMaps, kernel_size=KN.convTranspose, strides=(2,2), padding=padding, activation=AC.layers, trainable=trainable)(WBp)
                UP = KLayers.merge.concatenate( [WBp, contracting_Info[nL+1]] , axis=3)
                conv = Layer(featureMaps, trainable, UP)
                conv = Layer(featureMaps, trainable, conv)               
                if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers -1)):  
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(WB, nL):
            trainable = False if TF.Mode and nL in TF.FrozenLayers else True
            featureMaps = FM*(2**nL)

            WB = Layer(featureMaps, trainable, WB)
            WB = Layer(featureMaps, trainable, WB)
            if DT.Mode and trainable: WB = KLayers.Dropout(DT.Value)(WB)
            return WB
                
        inputs = KLayers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB , NLayers-1)

        WB = Unet_sublayer_Expanding(WB , Conv_Out)

        final = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    """
    def CNN_Classifier(ModelParam):
        dim   = ModelParam.Method.InputImage2Dvs3D
        # input_shape= tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
        model = kerasmodels.Sequential()
        model.add(KLayers.Conv2D(filters=16, kernel_size=ModelParam.kernel_size, padding=ModelParam.padding, activation=ModelParam.activitation, input_shape=input_shape  ))
        model.add(KLayers.MaxPooling2D(pool_size=ModelParam.Layer_Params.MaxPooling.pool_size))
        model.add(KLayers.Dropout(ModelParam.Layer_Params.Dropout.Value))

        model.add(KLayers.Conv2D(filters=8, kernel_size=ModelParam.kernel_size, padding=ModelParam.padding, activation=ModelParam.activitation, input_shape=input_shape ))
        model.add(KLayers.MaxPooling2D(pool_size=ModelParam.Layer_Params.MaxPooling.pool_size))
        model.add(KLayers.Dropout(ModelParam.Layer_Params.Dropout.Value))

        model.add(KLayers.Flatten())
        model.add(KLayers.Dense(128 , activation=ModelParam.Layer_Params.Activitation.layers))
        model.add(KLayers.Dropout(ModelParam.Layer_Params.Dropout.Value))
        model.add(KLayers.Dense(ModelParam.MultiClass.num_classes , activation=ModelParam.Layer_Params.Activitation.output))

        return model

    def FCN_3D(ModelParam):

        NumberFM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num
        
        # input_shape = tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
        inputs = KLayers.Input( input_shape )

        conv = inputs
        for nL in range(ModelParam.num_Layers -1):
            Trainable = False if ModelParam.Transfer_Learning.Mode and nL in ModelParam.Transfer_Learning.FrozenLayers else True
            conv = KLayers.Conv3D(filters=NumberFM*(2**nL), kernel_size=ModelParam.Layer_Params.ConvLayer.Kernel_size.conv, padding=ModelParam.Layer_Params.ConvLayer.padding, activation=ModelParam.Layer_Params.Activitation.layers, trainable=Trainable)(conv)
            if ModelParam.Layer_Params.Dropout.Mode and Trainable: conv = KLayers.Dropout(ModelParam.Layer_Params.Dropout)(conv)

        final  = KLayers.Conv3D(filters=ModelParam.MultiClass.num_classes, kernel_size=ModelParam.Layer_Params.ConvLayer.Kernel_size.output, padding=ModelParam.Layer_Params.ConvLayer.padding, activation=ModelParam.Layer_Params.Activitation.output)(conv)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    def FCN_2D(ModelParam):

        NumberFM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num
        # input_shape = tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
        inputs = KLayers.Input( input_shape )

        conv = inputs
        for nL in range(ModelParam.num_Layers -1):
            Trainable = False if ModelParam.Transfer_Learning.Mode and nL in ModelParam.Transfer_Learning.FrozenLayers else True
            conv = KLayers.Conv2D(filters=NumberFM*(2**nL), kernel_size=ModelParam.Layer_Params.ConvLayer.Kernel_size.conv, padding=ModelParam.Layer_Params.ConvLayer.padding, activation=ModelParam.Layer_Params.Activitation.layers, trainable=Trainable)(conv)
            if ModelParam.Layer_Params.Dropout.Mode and Trainable: conv = KLayers.Dropout(ModelParam.Layer_Params.Dropout)(conv)

        final  = KLayers.Conv2D(filters=ModelParam.MultiClass.num_classes, kernel_size=ModelParam.Layer_Params.ConvLayer.Kernel_size.output, padding=ModelParam.Layer_Params.ConvLayer.padding, activation=ModelParam.Layer_Params.Activitation.output)(conv)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    """    
    if  ModelParam.architectureType == 'U-Net':
        model = UNet(ModelParam)
    
    elif  ModelParam.architectureType == 'U-Net4':
        model = UNet4(ModelParam)

    elif  ModelParam.architectureType == 'FCN':
        model = FCN(ModelParam)

    elif  ModelParam.architectureType == 'FCN_with_SkipConnection':
        model = FCN_with_SkipConnection(ModelParam)

    model.summary()

    return model

def func_classWeights(params , Data):

    if params.WhichExperiment.HardParams.Model.Layer_Params.class_weight.Mode:
        Sz = Data.Train.Mask.shape
        a = np.sum(Data.Train.Mask[...,0] < 0.5) / (Sz[0]*Sz[1]*Sz[2])
        return {0:1-a , 1:a}
    else: return {0:1 , 1:1}
   
class DataGenerator_Class(keras.utils.Sequence):

    # ! source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    
    # Note : batch_size in this class means the number of subjects rather than number of 2D images 

    def __init__(self, params , mode = 'train' , shuffle=True):
        
        
        def func_list_subjects(self):
            TrainList, ValList = datasets.percentageDivide(self.params.WhichExperiment.Dataset.Validation.percentage, list(self.params.directories.Train.Input.Subjects), self.params.WhichExperiment.Dataset.randomFlag)            

            with h5py.File(self.address + '/Data.hdf5','r') as f: 
                switcher = {
                    'train':      list(f['train']) if self.TestForVal else TrainList, 
                    'validation': list(f['test'])  if self.TestForVal else ValList, 
                }
                return switcher.get(self.mode, 'wrong mode')
        self.params     = params
        self.TestForVal = self.params.WhichExperiment.HardParams.Model.Method.Use_TestCases_For_Validation
        self.shuffle    = shuffle
        self.mode       = mode
        if mode == 'validation': self.DataMode = 'test' if self.TestForVal else 'train'
        else: self.DataMode = mode

        
        self.address = self.params.directories.Test.Result
        self.batch_size = self.params.WhichExperiment.HardParams.Model.DataGenerator.NumSubjects_Per_batch
        self.list_subjects = func_list_subjects(self)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_subjects) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        list_subjects_oneBatch = self.list_subjects[index*self.batch_size:(index+1)*self.batch_size]

        return self.__data_generation(list_subjects_oneBatch)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True: np.random.shuffle(self.list_subjects)

    def __data_generation(self, list_subjects_oneBatch):
        # Generate data
        with h5py.File(self.address + '/Data.hdf5','r') as Data:
            for cnt , ID in enumerate(list_subjects_oneBatch):
                
                im  = np.array(list(  Data['%s/%s/Image'%(self.DataMode,ID)] ))  
                msk = np.array(list(  Data['%s/%s/Mask'%( self.DataMode,ID)] ))

                Image = im if cnt == 0 else np.concatenate((Image , im) , axis=0)
                Mask = msk if cnt == 0 else np.concatenate((Mask , msk) , axis=0)

        return Image , Mask
