import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras import models as kerasmodels
import numpy as np
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
from tqdm import tqdm
import nibabel as nib
from scipy import ndimage
import pickle
import keras
from keras.utils import multi_gpu_model
import keras.layers as keras_layers
import modelFuncs.LossFunction as LossFunction


def func_class_weights(Mask):
    """ Finding the weights for each class, in an unbalanced dataset

        Args:
            Mask: A 4D volume of all inputs and their labels. Dimension: (n = 2D slices , x , y , c = classes)
        
        Returns:
            class weights (numpy array): 
    """

    sz = Mask.shape
    NUM_CLASSES = sz[3]
    NUM_SAMPLES = np.prod(sz[:3])

    class_weights = np.ones(NUM_CLASSES)

    for ix in range(NUM_CLASSES):
        TRUE_Count = len(np.where(Mask[..., ix] > 0.5)[0])
        class_weights[ix] = NUM_SAMPLES / (NUM_CLASSES * TRUE_Count)

    class_weights = class_weights / np.sum(class_weights)

    print('class_weights', class_weights)

    return class_weights


def check_Run(params, Data):
    # Assigning the gpu index
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index

    if params.WhichExperiment.TestOnly.mode or params.UserInfo['thalamic_side'].active_side == 'right':
        # Skipping the training phase, if the algorithm is set to test from existing trained networks or the right thalamus
        model = loadModel(params)

    else:
        # Training the network, if the algorithm is set to training on the left thalamus
        model = trainingExperiment(Data, params)

    # predicting the labels on test cases
    prediction = testingExeriment(model, Data, params)

    # Saving the predicted whole thalamus bounding box as a text file
    nucleus_index = int(params.WhichExperiment.Nucleus.Index[0])
    if nucleus_index == 1:
        save_BoundingBox_Hierarchy(params, prediction)

    return True


def loadModel(params):
    """ Loading the model

        Args:
            params: Parameters

        Returns:
            model: neural network model
    """

    # Loading the architecture: 
    #    To bypass the dependency on the number of index of gpu, weights are loaded separately than the architecture
    model = architecture(params.WhichExperiment.HardParams.Model)

    # Loading the weights
    model.load_weights(params.directories.Train.Model + '/model_weights.h5')

    return model


def testingExeriment(model, Data, params):
    class prediction:
        Test = ''
        Train = ''

    def predictingTestSubject(DataSubj, subject, ResultDir):

        # params.directories.Test.Input.Subjects
        def postProcessing(pred1Class, origMsk1N, NucleiIndex):
            """ Post Processing

            Args:
                pred1Class (numpy array): Prediction mask for 1 nucleus
                origMsk1N  (numpy array): Manual label mask for 1 nucleus
                NucleiIndex        (int): Nucleus index
            """

            def binarizing(pred1N):
                """ Binarizing the input mask """

                # Thresh = max( skimage.filters.threshold_otsu(pred1N) ,0.2)  if len(np.unique(pred1N)) != 1 else 0
                return pred1N > 0.5

            def cascade_paddingToOrigSize(im):
                """ Removing the added padding prior to savin the inputs """

                if 1 not in params.WhichExperiment.Nucleus.Index:
                    im = np.pad(im, subject.NewCropInfo.PadSizeBackToOrig, 'constant')
                return im

            def closeMask(mask):
                """ Applying morphology filters onto the prediction masks """

                struc = ndimage.generate_binary_structure(3, 2)
                return ndimage.binary_closing(mask, structure=struc)

            # Removig the extra dimension
            pred = np.squeeze(pred1Class)

            # Binarizing the predicted segmentation mask
            pred2 = binarizing(pred)

            # Applying morphology filters onto the prediction mask
            if params.WhichExperiment.HardParams.Model.Method.ImClosePrediction:
                pred2 = closeMask(pred2)

            # Extracting the most dominant object (biggest object) inside the prediction mask
            pred2 = smallFuncs.extracting_the_biggest_object(pred2)

            if subject.Label.address:
                # Re-orienting the manual label mask into the network's orientation
                label_mask = np.transpose(origMsk1N, params.WhichExperiment.Dataset.slicingInfo.slicingOrder)

                # Measuring Dice value
                Dice = [NucleiIndex, smallFuncs.mDice(pred2, binarizing(label_mask))]
            else:
                Dice = [0, 0.0]

            # Removing the extra paddings added to the input prior to feeding to the network
            pred = cascade_paddingToOrigSize(pred2)

            # Re-orienting the prediction mask into its original orientation (Axial -> Sagittal -> Coronal)
            pred = np.transpose(pred, params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse)

            return pred, Dice

        def savingOutput(pred1N_BtO, NucleiIndex):
            dirSave = smallFuncs.mkDir(ResultDir)
            nucleus = smallFuncs.Nuclei_Class(index=NucleiIndex).name
            smallFuncs.saveImage(pred1N_BtO, DataSubj.Affine, DataSubj.Header, dirSave + '/' + nucleus + '.nii.gz')
            return dirSave, nucleus

        def applyPrediction():

            # Number of classes
            num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes

            # If the background is added as an extra input, the overall number of classes will be the number of nuclei + 1
            if not params.WhichExperiment.HardParams.Model.Method.havingBackGround_AsExtraDimension:
                num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes + 1

            def loopOver_AllClasses_postProcessing(pred):

                Dice = np.zeros((num_classes - 1, 2))
                ALL_pred = []
                for cnt in range(num_classes - 1):

                    nucleus_index = params.WhichExperiment.Nucleus.Index[cnt]

                    # Manual Label for nucleus [cnt: index]
                    manual_label = np.array([])
                    if subject.Label.address: manual_label = DataSubj.OrigMask[..., cnt]

                    pred1N_BtO, Dice[cnt, :] = postProcessing(pred[..., cnt], manual_label, nucleus_index)

                    # If the first cascade network (on whole thalamus) is running, this concatenates the prediciton 
                    # masks for the following step of saving the predicted whole thelamus encompassing boundingbox
                    if int(params.WhichExperiment.Nucleus.Index[0]) == 1:
                        # if cnt > 0:
                        #     ALL_pred = np.concatenate((ALL_pred, pred1N_BtO[..., np.newaxis]), axis=3) 
                        # else:
                        ALL_pred = pred1N_BtO[..., np.newaxis]

                    dirSave, nucleusName = savingOutput(pred1N_BtO, params.WhichExperiment.Nucleus.Index[cnt])

                # Saving all nuclei Dices into one text file
                if num_classes > 2 and params.WhichExperiment.HardParams.Model.MultiClass.Mode:
                    Dir_Dice = dirSave + '/Dice_All.txt'

                    # Saving the Dice value for the predicted nucleus
                else:
                    Dir_Dice = dirSave + '/Dice_' + nucleusName + '.txt'

                # Saving the Dice values
                np.savetxt(Dir_Dice, Dice, fmt='%1.1f %1.4f')
                return ALL_pred

            def unPadding(im, pad):
                sz = im.shape
                if np.min(pad) < 0:
                    pad, crd = datasets.paddingNegativeFix(sz, pad)
                    for ix in range(3):
                        crd[ix, 1] = 0 if crd[ix, 1] == sz[ix] else -crd[ix, 1]

                    # if len(crd) == 3: crd = np.append(crd,[0,0],axs=1)
                    crd = tuple([tuple(x) for x in crd])

                    im = im[pad[0][0]:sz[0] - pad[0][1], pad[1][0]:sz[1] - pad[1][1], pad[2][0]:sz[2] - pad[2][1], :]
                    im = np.pad(im, crd, 'constant')
                else:
                    im = im[pad[0][0]:sz[0] - pad[0][1], pad[1][0]:sz[1] - pad[1][1], pad[2][0]:sz[2] - pad[2][1], :]

                return im

            im = DataSubj.Image.copy()

            # Segmenting the test case using trained network
            predF = model.predict(im)

            # score = model.evaluate(DataSubj.Image, DataSubj.Mask)

            # Removing the extra background dimention
            pred = predF[..., :num_classes - 1]
            if len(pred.shape) == 3:
                pred = np.expand_dims(pred, axis=3)

            # Re-orienting the predicted segmentation masks into its original 3D volume dimentionality
            pred = np.transpose(pred, [1, 2, 0, 3])
            if len(pred.shape) == 3:
                pred = np.expand_dims(pred, axis=3)

            # Removing the extra voxels added to the input for the purpose of conforming to NN input dimensioanlity
            pred = unPadding(pred, subject.Padding)

            # Running the post processing on all classes
            return loopOver_AllClasses_postProcessing(pred)

        return applyPrediction()

    def loopOver_Predicting_TestSubjects(DataTest):
        prediction = {}
        # ResultDir = params.directories.Test.Result
        for name in tqdm(DataTest, desc='predicting test subjects'):
            subject = params.directories.Test.Input.Subjects[name]
            ResultDir = subject.address + '/' + params.UserInfo['thalamic_side'].active_side + '/sd' + str(
                params.WhichExperiment.Dataset.slicingInfo.slicingDim) + '/'
            prediction[name] = predictingTestSubject(DataTest[name], subject, ResultDir)

        return prediction

    def loopOver_Predicting_TestSubjects_Sagittal(DataTest):
        prediction = {}
        # ResultDir = params.directories.Test.Result.replace('/sd2','/sd0')
        for name in tqdm(DataTest, desc='predicting test subjects sagittal'):
            subject = params.directories.Test.Input.Subjects[name]
            ResultDir = subject.address + '/' + params.UserInfo['thalamic_side'].active_side + '/sd0/'
            prediction[name] = predictingTestSubject(DataTest[name], subject, ResultDir)

        return prediction

    def loopOver_Predicting_TrainSubjects(DataTrain):

        TestOnly = params.WhichExperiment.TestOnly.mode
        Measure_Dice_on_Train_Data = params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data
        thalamus_network = (int(params.WhichExperiment.Nucleus.Index[0]) == 1)

        prediction = {}
        if (not TestOnly) and (Measure_Dice_on_Train_Data or thalamus_network):

            ResultDir = smallFuncs.mkDir(params.directories.Test.Result + '/TrainData_Output')
            for name in tqdm(DataTrain, desc='predicting train subjects'):
                subject = params.directories.Train.Input.Subjects[name]
                prediction[name] = predictingTestSubject(DataTrain[name], subject,
                                                         ResultDir + '/' + subject.subjectName + '/')

        return prediction

    def loopOver_Predicting_TrainSubjects_Sagittal(DataTrain):

        TestOnly = params.WhichExperiment.TestOnly.mode
        Measure_Dice_on_Train_Data = params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data
        thalamus_network = (int(params.WhichExperiment.Nucleus.Index[0]) == 1)

        prediction = {}
        if (not TestOnly) and (Measure_Dice_on_Train_Data or thalamus_network):

            ResultDir = smallFuncs.mkDir(params.directories.Test.Result.replace('/sd2', '/sd0') + '/TrainData_Output')
            for name in tqdm(DataTrain, desc='predicting train subjects sagittal'):
                subject = params.directories.Train.Input.Subjects[name]
                prediction[name] = predictingTestSubject(DataTrain[name], subject,
                                                         ResultDir + '/' + subject.subjectName + '/')

        return prediction

    prediction.Test = loopOver_Predicting_TestSubjects(Data.Test)
    prediction.Train = loopOver_Predicting_TrainSubjects(Data.Train_ForTest)

    if params.WhichExperiment.Nucleus.Index[0] == 1 and params.WhichExperiment.Dataset.slicingInfo.slicingDim == 2:
        prediction.Sagittal_Test = loopOver_Predicting_TestSubjects_Sagittal(Data.Sagittal_Test)
        prediction.Sagittal_Train = loopOver_Predicting_TrainSubjects_Sagittal(Data.Sagittal_Train_ForTest)

    return prediction


def trainingExperiment(Data, params):
    def func_CallBacks(params):
        Dir_Save = params.directories.Train.Model
        mode = 'max'
        monitor = 'val_mDice'

        checkpointer = keras.callbacks.ModelCheckpoint(filepath=Dir_Save + '/best_model_weights.h5', \
                                                       monitor='val_mDice', verbose=1, save_best_only=True, mode=mode)

        Reduce_LR = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, min_delta=0.001, patience=15,
                                                      verbose=1, save_best_only=True, mode=mode, min_lr=1e-6, )

        def step_decay_schedule(initial_lr=params.WhichExperiment.HardParams.Model.Learning_Rate, decay_factor=0.5,
                                step_size=18):
            def schedule(epoch):
                return initial_lr * (decay_factor ** np.floor(epoch / step_size))

            return keras.callbacks.LearningRateScheduler(schedule, verbose=1)

        EarlyStopping = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=40, verbose=1, mode=mode, \
                                                      baseline=0, restore_best_weights=True)

        if params.WhichExperiment.HardParams.Model.LR_Scheduler:
            return [checkpointer, EarlyStopping, Reduce_LR]
        else:
            return [checkpointer, EarlyStopping]  # TQDMCallback()

    def saveReport(DirSave, name, data):

        def savePickle(Dir, data):
            f = open(Dir, "wb")
            pickle.dump(data, f)
            f.close()

        savePickle(DirSave + '/' + name + '.pkl', data)

    def modelTrain_Unet(Data, params, modelS):
        ModelParam = params.WhichExperiment.HardParams.Model

        def modelInitialize(model):

            # Number of featuremaps in the first layer of ResUnet
            FM = '/FM' + str(params.WhichExperiment.HardParams.Model.Layer_Params.FirstLayer_FeatureMap_Num)

            # Nucleus name
            NN = '/' + params.WhichExperiment.Nucleus.name

            # Image orientation
            SD = '/sd' + str(params.WhichExperiment.Dataset.slicingInfo.slicingDim)

            initialization = params.WhichExperiment.HardParams.Model.Initialize

            # Address to the code
            code_address = smallFuncs.dir_check(params.UserInfo['experiment'].code_address)

            if initialization.init_address:
                init_address = smallFuncs.dir_check(initialization.init_address) + FM + NN + SD + '/model_weights.h5'
            else:
                # The modality of the input image that is defined by the user
                modDef = params.WhichExperiment.Experiment.image_modality.lower()

                # If the input modality is set to WMn, the network trained on SRI dataset will be used for initialization
                if modDef == 'wmn':
                    net_name = 'sri'

                    # If the input modality is set to CSFn, the network trained on WMn dataset will be used for initialization
                elif modDef == 'csfn':
                    net_name = 'wmn'

                # The address to initialization network based on the number of featuremaps, nucleus name and image orientation
                init_address = code_address + 'Trained_Models/' + net_name + FM + NN + SD + '/model_weights.h5'

            try:
                model.load_weights(init_address)
                print(' --- initialization succesfull')
            except:
                print(' --- initialization failed')

            return model

        model = modelInitialize(modelS)

        if len(params.WhichExperiment.HardParams.Machine.GPU_Index) > 1: model = multi_gpu_model(model)

        def saveModel_h5(model, modelS):
            smallFuncs.mkDir(params.directories.Train.Model)
            if params.WhichExperiment.HardParams.Model.Method.save_Best_Epoch_Model:
                model.load_weights(params.directories.Train.Model + '/best_model_weights.h5')

            model.save(params.directories.Train.Model + '/orig_model.h5', overwrite=True, include_optimizer=False)
            model.save_weights(params.directories.Train.Model + '/orig_model_weights.h5', overwrite=True)

            modelS.save(params.directories.Train.Model + '/model.h5', overwrite=True, include_optimizer=False)
            modelS.save_weights(params.directories.Train.Model + '/model_weights.h5', overwrite=True)

            return model

        def modelFit(model):

            class_weights = func_class_weights(Data.Train.Mask)
            _, loss_tag = LossFunction.LossInfo(params.UserInfo['simulation'].lossFunction_Index)

            if 'My' in loss_tag:
                model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss(class_weights),
                              metrics=ModelParam.metrics)
            else:
                model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss, metrics=ModelParam.metrics)

            def func_modelParams():
                callbacks = func_CallBacks(params)

                batch_size = params.WhichExperiment.HardParams.Model.batch_size
                epochs = params.WhichExperiment.HardParams.Model.epochs
                valSplit_Per = params.WhichExperiment.Dataset.Validation.percentage
                verbose = params.WhichExperiment.HardParams.Model.verbose
                Validation_fromKeras = params.WhichExperiment.Dataset.Validation.fromKeras

                return callbacks, batch_size, epochs, valSplit_Per, verbose, Validation_fromKeras

            callbacks, batch_size, epochs, valSplit_Per, verbose, Validation_fromKeras = func_modelParams()

            def func_without_Generator():
                if params.WhichExperiment.HardParams.Model.Layer_Params.class_weight.Mode and ('My' not in loss_tag):
                    hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, class_weight=class_weights,
                                     batch_size=batch_size, epochs=epochs, shuffle=True,
                                     validation_data=(Data.Validation.Image, Data.Validation.Mask), verbose=verbose,
                                     callbacks=callbacks)  # , callbacks=[TQDMCallback()])
                else:
                    hist = model.fit(x=Data.Train.Image, y=Data.Train.Mask, batch_size=batch_size, epochs=epochs,
                                     shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Mask),
                                     verbose=verbose, callbacks=callbacks)  # , callbacks=[TQDMCallback()])

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
    saveReport(params.directories.Train.Model, 'hist_history', hist.history)
    return model


def save_BoundingBox_Hierarchy(params, PRED):
    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)

    def save_BoundingBox(PreStageMask, subject, directory):

        def checkBordersOnBoundingBox(Sz, BB, gapS):
            return [[np.max([BB[d][0] - gapS, 0]), np.min([BB[d][1] + gapS, Sz[d]])] for d in range(3)]

        imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
        # if 'train' in mode: 
        #     directory += '/TrainData_Output'

        for ch in range(PreStageMask.shape[3]):
            BB = smallFuncs.findBoundingBox(PreStageMask[..., ch])
            gapDilation = params.WhichExperiment.Dataset.gapDilation
            BBd = [[BB[ii][0] - gapDilation, BB[ii][1] + gapDilation] for ii in range(len(BB))]
            BBd = checkBordersOnBoundingBox(imF.shape, BBd, 0)

            BB = checkBordersOnBoundingBox(imF.shape, BB, params.WhichExperiment.Dataset.gapOnSlicingDimention)

            nucleusName = smallFuncs.Nuclei_Class(index=params.WhichExperiment.Nucleus.Index[ch]).name
            np.savetxt(directory + '/BB_' + nucleusName + '.txt', np.concatenate((BB, BBd), axis=1), fmt='%d')

    nucleus = params.WhichExperiment.Nucleus.name

    def loop_Subjects(PRED, mode):
        if PRED:
            if 'train' in mode:
                Subjects = params.directories.Train.Input.Subjects
                for name in tqdm(Subjects, desc='saving BB ' + ' ' + mode + nucleus):
                    save_BoundingBox(PRED[name], Subjects[name],
                                     params.directories.Test.Result + '/TrainData_Output/' + name + '/')

            elif 'test' in mode:
                Subjects = params.directories.Test.Input.Subjects
                for name in tqdm(Subjects, desc='saving BB ' + ' ' + mode + nucleus):
                    ResultDir = Subjects[name].address + '/' + params.UserInfo[
                        'thalamic_side'].active_side + '/sd' + str(
                        params.WhichExperiment.Dataset.slicingInfo.slicingDim) + '/'
                    save_BoundingBox(PRED[name], Subjects[name], ResultDir)

    def loop_Subjects_Sagittal(PRED, mode):
        if PRED:

            if 'train' in mode:
                Subjects = params.directories.Train.Input.Subjects
                for name in tqdm(Subjects, desc='saving BB ' + ' ' + mode + nucleus):
                    save_BoundingBox(PRED[name], Subjects[name], params.directories.Test.Result.replace('/sd2',
                                                                                                        '/sd0') + '/TrainData_Output/' + name)

            elif 'test' in mode:
                Subjects = params.directories.Test.Input.Subjects
                for name in tqdm(Subjects, desc='saving BB ' + ' ' + mode + nucleus):
                    ResultDir = Subjects[name].address + '/' + params.UserInfo['thalamic_side'].active_side + '/sd0/'
                    save_BoundingBox(PRED[name], Subjects[name], ResultDir)

    loop_Subjects(PRED.Test, 'test')
    loop_Subjects(PRED.Train, 'train')

    if params.WhichExperiment.Nucleus.Index[0] == 1 and params.WhichExperiment.Dataset.slicingInfo.slicingDim == 2:
        loop_Subjects_Sagittal(PRED.Sagittal_Test, 'test')
        loop_Subjects_Sagittal(PRED.Sagittal_Train, 'train')


def architecture(ModelParam):
    input_shape = tuple(ModelParam.InputDimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)

    def Res_Unet(ModelParam):  # Conv -> BatchNorm -> Relu ) -> (Conv -> BatchNorm -> Relu)  -> maxpooling  -> Dropout

        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num

        padding = ModelParam.Layer_Params.ConvLayer.padding
        NLayers = ModelParam.num_Layers
        num_classes = ModelParam.MultiClass.num_classes
        pool_size = ModelParam.Layer_Params.MaxPooling.pool_size

        def Layer(featureMaps, trainable, input_layer):
            conv = keras_layers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(
                input_layer)
            conv = keras_layers.BatchNormalization()(conv)
            return keras_layers.Activation(AC.layers)(conv)

        def Unet_sublayer_Contracting(inputs):
            def main_USC(WBp, nL):
                trainable = True
                featureMaps = FM * (2 ** nL)

                conv = Layer(featureMaps, trainable, WBp)
                conv = Layer(featureMaps, trainable, conv)

                # ! Residual Part
                conv = keras_layers.merge.concatenate([WBp, conv], axis=3)

                pool = keras_layers.MaxPooling2D(pool_size=pool_size)(conv)

                if trainable: pool = keras_layers.Dropout(DT.Value)(pool)

                return pool, conv

            for nL in range(NLayers - 1):
                if nL == 0: WB, Conv_Out = inputs, {}
                WB, Conv_Out[nL + 1] = main_USC(WB, nL)

            return WB, Conv_Out

        def Unet_sublayer_Expanding(WB, Conv_Out):
            def main_USE(WBp, nL, contracting_Info):
                trainable = True
                featureMaps = FM * (2 ** nL)

                WBp = keras_layers.Conv2DTranspose(featureMaps, kernel_size=KN.convTranspose, strides=(2, 2),
                                                   padding=padding, activation=AC.layers, trainable=trainable)(WBp)
                UP = keras_layers.merge.concatenate([WBp, contracting_Info[nL + 1]], axis=3)

                conv = Layer(featureMaps, trainable, UP)
                conv = Layer(featureMaps, trainable, conv)

                # ! Residual Part
                conv = keras_layers.merge.concatenate([UP, conv], axis=3)

                if DT.Mode and trainable: conv = keras_layers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers - 1)):
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(input_layer, nL):
            trainable = True
            featureMaps = FM * (2 ** nL)

            WB = Layer(featureMaps, trainable, input_layer)
            WB = Layer(featureMaps, trainable, WB)

            # ! Residual Part
            WB = keras_layers.merge.concatenate([input_layer, WB], axis=3)

            if DT.Mode and trainable: WB = keras_layers.Dropout(DT.Value)(WB)
            return WB

        inputs = keras_layers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB, NLayers - 1)

        WB = Unet_sublayer_Expanding(WB, Conv_Out)

        final = keras_layers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    def Res_Unet2(ModelParam):  # Conv -> BatchNorm -> Relu ) -> (Conv -> BatchNorm -> Relu)  -> maxpooling  -> Dropout

        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num

        padding = ModelParam.Layer_Params.ConvLayer.padding
        NLayers = ModelParam.num_Layers
        num_classes = ModelParam.MultiClass.num_classes
        pool_size = ModelParam.Layer_Params.MaxPooling.pool_size

        def Layer(featureMaps, trainable, input_layer):
            conv = keras_layers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(
                input_layer)
            conv = keras_layers.BatchNormalization()(conv)
            return keras_layers.Activation(AC.layers)(conv)

        def Unet_sublayer_Contracting(inputs):
            def main_USC(WBp, nL):
                trainable = True
                featureMaps = FM * (2 ** nL)

                conv = Layer(featureMaps, trainable, WBp)
                conv = Layer(featureMaps, trainable, conv)

                # ! Residual Part
                if nL > 0: conv = keras_layers.merge.concatenate([WBp, conv], axis=3)

                pool = keras_layers.MaxPooling2D(pool_size=pool_size)(conv)

                if trainable: pool = keras_layers.Dropout(DT.Value)(pool)

                return pool, conv

            for nL in range(NLayers - 1):
                if nL == 0: WB, Conv_Out = inputs, {}
                WB, Conv_Out[nL + 1] = main_USC(WB, nL)

            return WB, Conv_Out

        def Unet_sublayer_Expanding(WB, Conv_Out):
            def main_USE(WBp, nL, contracting_Info):
                trainable = True
                featureMaps = FM * (2 ** nL)

                WBp = keras_layers.Conv2DTranspose(featureMaps, kernel_size=KN.convTranspose, strides=(2, 2),
                                                   padding=padding, activation=AC.layers, trainable=trainable)(WBp)
                UP = keras_layers.merge.concatenate([WBp, contracting_Info[nL + 1]], axis=3)

                conv = Layer(featureMaps, trainable, UP)
                conv = Layer(featureMaps, trainable, conv)

                # ! Residual Part
                conv = keras_layers.merge.concatenate([UP, conv], axis=3)

                if DT.Mode and trainable: conv = keras_layers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers - 1)):
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(input_layer, nL):
            trainable = True
            featureMaps = FM * (2 ** nL)

            WB = Layer(featureMaps, trainable, input_layer)
            WB = Layer(featureMaps, trainable, WB)

            # ! Residual Part
            WB = keras_layers.merge.concatenate([input_layer, WB], axis=3)

            if DT.Mode and trainable: WB = keras_layers.Dropout(DT.Value)(WB)
            return WB

        inputs = keras_layers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB, NLayers - 1)

        WB = Unet_sublayer_Expanding(WB, Conv_Out)

        final = keras_layers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    if ModelParam.architectureType == 'Res_Unet':
        model = Res_Unet(ModelParam)
    elif ModelParam.architectureType == 'Res_Unet2':
        model = Res_Unet2(ModelParam)

    # model.summary()

    return model
