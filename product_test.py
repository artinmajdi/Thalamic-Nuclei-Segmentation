import os
import sys
sys.path.append(os.path.dirname(__file__))  # sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
from otherFuncs.datasets import preAnalysis
from otherFuncs import datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess
import tensorflow as tf
from keras import backend as K
import pandas as pd
import xlsxwriter
import csv
import numpy as np
import nibabel as nib
from tqdm import tqdm
import skimage
from scipy import ndimage
import keras
from keras.utils import multi_gpu_model, multi_gpu_utils
import h5py
import keras.layers as KLayers
from preprocess import BashCallingFunctionsA, croppingA
from keras import models as kerasmodels
from skimage.transform import AffineTransform , warp
import Parameters.UserInfo as UserInfo


def terminalEnteries(UserInfo):

    def func_slicingDim(sysargv):
        if sysargv.lower() == 'all':  return [2,1,0]
        elif sysargv[0] == '[':       return [int(k) for k in sysargv.split('[')[1].split(']')[0].split(",")]
        else:                         return [int(sysargv)]

    UserInfo['Input'].left = True
    UserInfo['Input'].orientation = [2,1,0]

    for en in range(len(sys.argv)):
        entry = sys.argv[en]

        if entry.lower() in ('-g','--gpu'): 
            UserInfo['simulation'].GPU_Index = sys.argv[en+1]

        elif entry.lower() in ('-sd','--orientation'):
            UserInfo['Input'].orientation = func_slicingDim(sys.argv[en+1])

        elif (entry.lower() == '-wmn') or (entry.lower() == '-csfn'):
            UserInfo['Input'].directory = sys.argv[en+1]
            UserInfo['Input'].modality  = entry.lower().split('-')[1]

        elif entry in ('--no-cropping'):
            UserInfo['preprocess'].Cropping = False               
                    
        elif entry in ('--no-bias-correction'):
            UserInfo['preprocess'].BiasCorrection = False   

        elif entry in ('--no-bias-reslicing'):
            UserInfo['preprocess'].Reslicing = False   

        elif entry in ('--right'):
            UserInfo['Input'].right = True

        return UserInfo

    return UserInfo


def loadModel(params):
    model = architecture(params.WhichExperiment.HardParams.Model)
    A = params.directories.Train.model_Tag if params.WhichExperiment.HardParams.Model.Transfer_Learning.Mode else ''
    model.load_weights(params.directories.Train.Model + '/model_weights' + A + '.h5')
    
    # model = kerasmodels.load_model(params.directories.Train.Model + '/model.h5')
    # ModelParam = params.WhichExperiment.HardParams.Model
    # model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss , metrics=ModelParam.metrics)
    return model

def architecture(ModelParam):

    if ModelParam.Upsample.Mode:
        scale , szI = ModelParam.Upsample.Scale   ,  ModelParam.InputDimensions        
        InDim_new = (scale*szI[0] , scale*szI[1] , szI[2] , 1)
        input_shape = tuple(InDim_new[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
    else:
        input_shape = tuple(ModelParam.InputDimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)

    def UNet4(ModelParam):
                    
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
                trainable = TF.U_Net4.Contracting[nL] if TF.Mode else True
                featureMaps = FM*(2**nL)

                conv = Layer(featureMaps, trainable, WBp)
                conv = Layer(featureMaps, trainable, conv)
                
                pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)                                
                
                # if DT.Mode and trainable: pool = KLayers.Dropout(DT.Value)(pool)  
                pool = KLayers.Dropout(DT.Value)(pool)  
                                
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
                
                # if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
                conv = KLayers.Dropout(DT.Value)(conv)
                return conv

            for nL in reversed(range(NLayers -1)):  
                WB = main_USE(WB, nL, Conv_Out)

            return WB

        def Unet_MiddleLayer(WB, nL):
            trainable = TF.U_Net4.Middle if TF.Mode else True
            featureMaps = FM*(2**nL)

            WB = Layer(featureMaps, trainable, WB)
            WB = Layer(featureMaps, trainable, WB)     
            WB = KLayers.Dropout(DT.Value)(WB)
            
            return WB
                
        inputs = KLayers.Input(input_shape)

        WB, Conv_Out = Unet_sublayer_Contracting(inputs)

        WB = Unet_MiddleLayer(WB , NLayers-1)

        WB = Unet_sublayer_Expanding(WB , Conv_Out)

        final = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(WB)

        return kerasmodels.Model(inputs=[inputs], outputs=[final])

    def Res_Unet2(ModelParam):
                    
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
                trainable = TF.U_Net4.Contracting[nL] if TF.Mode else True
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

    def ResFCN_ResUnet2_TL(ModelParam):

        TF = ModelParam.Transfer_Learning
        LP = ModelParam.Layer_Params        
        KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
        AC = ModelParam.Layer_Params.Activitation
        DT = ModelParam.Layer_Params.Dropout
        FM_FCN = ModelParam.Layer_Params.FCN_FeatureMaps

        padding      = ModelParam.Layer_Params.ConvLayer.padding        
        FCN1_NLayers = ModelParam.FCN1_NLayers
        FCN2_NLayers = ModelParam.FCN2_NLayers
        num_classes  = ModelParam.MultiClass.num_classes
        pool_size    = ModelParam.Layer_Params.MaxPooling.pool_size
        FM      = ModelParam.Best_WMn_Model.FM # ModelParam.Layer_Params.FirstLayer_FeatureMap_Num
        NLayers = ModelParam.Best_WMn_Model.NL # ModelParam.num_Layers
        
        def Layer(featureMaps, trainable, input):
            conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(input)
            conv = KLayers.BatchNormalization()(conv)  
            return KLayers.Activation(AC.layers)(conv) 

        def Unet(Unet_input):  


            def Unet_sublayer_Contracting(inputs):
                def main_USC(WBp, nL):
                    trainable = TF.U_Net4.Contracting[nL] if TF.Mode else True
                    featureMaps = FM*(2**nL)

                    conv = Layer(featureMaps, trainable, WBp)
                    conv = Layer(featureMaps, trainable, conv)
                    
                    pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)                                
                    
                    # if DT.Mode and trainable: pool = KLayers.Dropout(DT.Value)(pool)  
                    pool = KLayers.Dropout(DT.Value)(pool)  
                                    
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
                    
                    # if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
                    conv = KLayers.Dropout(DT.Value)(conv)
                    return conv

                for nL in reversed(range(NLayers -1)):  
                    WB = main_USE(WB, nL, Conv_Out)

                return WB
                
            def Unet_MiddleLayer(WB, nL):
                trainable = TF.U_Net4.Middle if TF.Mode else True
                featureMaps = FM*(2**nL)

                WB = Layer(featureMaps, trainable, WB)
                WB = Layer(featureMaps, trainable, WB)     
                WB = KLayers.Dropout(DT.Value)(WB)  

                return WB
                                    
            WB, Conv_Out = Unet_sublayer_Contracting(Unet_input)
            WB = Unet_MiddleLayer(WB , NLayers-1)
            WB = Unet_sublayer_Expanding(WB , Conv_Out)

            return WB

        def ResUnet2(ResUnet_input):

            def Unet_sublayer_Contracting(inputs):
                def main_USC(WBp, nL):
                    trainable = TF.U_Net4.Contracting[nL] if TF.Mode else True
                    featureMaps = FM*(2**nL)

                    conv = Layer(featureMaps, trainable, WBp)
                    conv = Layer(featureMaps, trainable, conv)                                              
                    
                    #! Residual Part
                    if nL > 0: conv = KLayers.merge.concatenate( [WBp, conv] , axis=3)  

                    pool = KLayers.MaxPooling2D(pool_size=pool_size)(conv)                                
                    
                    pool = KLayers.Dropout(DT.Value)(pool)  
                                    
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

                    conv = KLayers.Dropout(DT.Value)(conv)
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


                WB = KLayers.Dropout(DT.Value)(WB)
                return WB
                    
            WB, Conv_Out = Unet_sublayer_Contracting(ResUnet_input)
            WB = Unet_MiddleLayer(WB , NLayers-1)
            WB = Unet_sublayer_Expanding(WB , Conv_Out)

            return WB
       
        def FCN_Layer(conv,num_Layers):

            for _ in range(num_Layers):
                conv = Layer(FM_FCN, True, conv)                     
                conv = KLayers.Dropout(DT.Value)(conv)                 

            return conv            

        def ResFCN_Layer(inputs,num_Layers):
            for nL in range(num_Layers):
                if nL == 0: conv = Layer(FM_FCN, True, inputs)
                else: conv = Layer(FM_FCN, True, conv)
                                     
                conv = KLayers.Dropout(DT.Value)(conv)  

            conv = KLayers.merge.concatenate( [inputs, conv] , axis=3) 

            return conv       

        inputs = KLayers.Input(input_shape)
       
        gap = 0
        if FCN1_NLayers > 0: 
            conv = ResFCN_Layer(inputs, FCN1_NLayers)
            gap  = FCN1_NLayers*4+1


        else: conv = inputs

        conv = ResUnet2(conv)

        if FCN2_NLayers > 0: conv = ResFCN_Layer(conv, FCN2_NLayers) 

        output = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(conv)
        modelNew = kerasmodels.Model(inputs=[inputs], outputs=[output])

        if not ModelParam.TestOnly:
            best_WMn_model = kerasmodels.load_model(ModelParam.Best_WMn_Model.address) # num_Layers 43
            print( 'ResNet model address' , ModelParam.Best_WMn_Model.address )
            for l in tqdm(range(2,len(best_WMn_model.layers)-1), 'loading the weights for Res Unet'):
                modelNew.layers[l+gap].set_weights(best_WMn_model.layers[l].get_weights())

        return modelNew

       
    if    ModelParam.architectureType == 'U-Net4':              model = UNet4(ModelParam)
    elif  ModelParam.architectureType == 'Res_Unet2':           model = Res_Unet2(ModelParam)        
    elif  ModelParam.architectureType == 'ResFCN_ResUnet2_TL':  model = ResFCN_ResUnet2_TL(ModelParam)    
            
    model.summary()

    return model


def predictingTestSubject(model, params, Data, subject, ResultDir):

    def postProcessing(pred1Class, origMsk1N, NucleiIndex):

        def binarizing(pred1N):
            Thresh = max( skimage.filters.threshold_otsu(pred1N) ,0.2)  if len(np.unique(pred1N)) != 1 else 0
            return pred1N  > Thresh

        def cascade_paddingToOrigSize(im):
            if 1 not in params.WhichExperiment.Nucleus.Index:
                im = np.pad(im, subject.NewCropInfo.PadSizeBackToOrig, 'constant')
            return im

        def closeMask(mask):
            struc = ndimage.generate_binary_structure(3,2)
            return ndimage.binary_closing(mask, structure=struc)
        
        pred1N = np.squeeze(pred1Class)
        
        pred = cascade_paddingToOrigSize(pred1N)
        pred = np.transpose(pred , params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse)

        pred_Binary = binarizing( pred)
        
        if params.WhichExperiment.HardParams.Model.Method.ImClosePrediction: 
            pred_Binary = closeMask(pred_Binary)

        Dice = [ NucleiIndex , smallFuncs.mDice(pred_Binary , binarizing(origMsk1N)) ]

        return pred_Binary, Dice

    def savingOutput(pred1N_BtO, NucleiIndex):
        dirSave = smallFuncs.mkDir(ResultDir + '/' + subject.subjectName)
        nucleusName, _ , _ = smallFuncs.NucleiSelection(NucleiIndex)

        smallFuncs.saveImage( pred1N_BtO , Data.Affine, Data.Header, dirSave + '/' + nucleusName + '.nii.gz')
        return dirSave, nucleusName

    def applyPrediction():

        num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes

        def loopOver_AllClasses_postProcessing(pred):

            Dice = np.zeros((num_classes-1,2))
            for cnt in range(num_classes-1):

                pred1N_BtO, Dice[cnt,:] = postProcessing(pred[...,cnt], DataSubj.OrigMask[...,cnt] , params.WhichExperiment.Nucleus.Index[cnt] )
                
                ALL_pred = np.concatenate( (ALL_pred , pred1N_BtO[...,np.newaxis]) , axis=3) if cnt > 0 else pred1N_BtO[...,np.newaxis]

                dirSave, nucleusName = savingOutput(pred1N_BtO, params.WhichExperiment.Nucleus.Index[cnt])
            
            Dir_Dice = dirSave + '/Dice_All.txt' if (params.WhichExperiment.HardParams.Model.MultiClass.Mode and num_classes > 2 ) else dirSave + '/Dice_' + nucleusName + '.txt'
            np.savetxt(Dir_Dice ,Dice,fmt='%1.1f %1.4f')
            return ALL_pred

        def unPadding(im , pad):
            sz = im.shape
            if np.min(pad) < 0:
                pad, crd = datasets.paddingNegativeFix(sz, pad)
                for ix in range(3):  crd[ix,1] = 0  if crd[ix,1] == sz[ix] else -crd[ix,1]

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

        predF = model.predict(Data.Image)
        # score = model.evaluate(Data.Image, Data.Mask)
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
    
def main(UserInfoB):
        
    def func_predict(UserInfoB):
        if not ( (0 in UserInfoB['simulation'].slicingDim) and (1 in UserInfoB['simulation'].nucleus_Index)  ):
            params = paramFunc.Run(UserInfoB, terminal=False)
            Data, params = datasets.loadDataset(params)   

            model = loadModel(params)
            subject = params.directories.Test.Input.Subjects[list(params.directories.Test.Input.Subjects)[0]]
            predictingTestSubject(model, params, Data, subject, ResultDir)


            K.clear_session()        
            
    UserInfoB['simulation'].nucleus_Index = [1]
    func_predict(UserInfoB)

    BB = smallFuncs.Nuclei_Class(1,'Cascade')
    UserInfoB['simulation'].nucleus_Index = BB.remove_Thalamus_From_List(BB.All_Nuclei().Indexes)
    func_predict(UserInfoB)

def EXP_CSFn_test_new_Cases(UserInfoB):
        
    UserInfoB['TypeExperiment'] = 11
    UserInfoB['Model_Method'] = 'Cascade' 
    UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL' # ''
    UserInfoB['lossFunction_Index'] = 4
    UserInfoB['simulation'].batch_size = 50
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20    
    UserInfoB['simulation'].FCN1_NLayers = 0
    UserInfoB['simulation'].FCN2_NLayers = 0  
    UserInfoB['simulation'].FCN_FeatureMaps = 0

    applyPreprocess.main(paramFunc.Run(UserInfoB, terminal=True), 'experiment')

    for x in [2,1,0]:
        UserInfoB['simulation'].slicingDim = [x]
        UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
        main(UserInfoB)

    smallFuncs.apply_MajorityVoting(paramFunc.Run(UserInfoB, terminal=False))

def EXP_WMn_test_new_Cases(UserInfoB):
    
    def predict_Thalamus_For_SD0(UserI):

        UserI['simulation'].slicingDim = [2]
        UserI['simulation'].nucleus_Index = [1]
        main(UserI)

        UserI['simulation'].slicingDim = [0]
        UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
        main(UserI)
    
    def merge_results_and_apply_25D(UserInfoB):

        UserInfoB['best_network_MPlanar'] = True
        params = paramFunc.Run(UserInfoB, terminal=True)
        Directory = params.WhichExperiment.Experiment.address + '/results'
        Output = 'sE12_Cascade_FM00_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a'
        os.system("mkdir %s; cd %s; mkdir sd0 sd1 sd2"%(Directory + '/' + Output, Directory + '/' + Output))
        os.system("cp -r %s/sE12_Cascade_FM40_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/sd0/vimp* %s/%s/sd0/"%(Directory, Directory, Output) )
        os.system("cp -r %s/sE12_Cascade_FM30_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/sd1/vimp* %s/%s/sd1/"%(Directory, Directory, Output) )
        os.system("cp -r %s/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/sd2/vimp* %s/%s/sd2/"%(Directory, Directory, Output) )
        
        smallFuncs.apply_MajorityVoting(params)

    UserInfoB['Model_Method'] = 'Cascade'
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['architectureType'] = 'Res_Unet2'
    UserInfoB['lossFunction_Index'] = 4
    UserInfoB['Experiments'].Index = '6'
    UserInfoB['TypeExperiment'] = 15
    UserInfoB['simulation'].LR_Scheduler = True    
    
    applyPreprocess.main(paramFunc.Run(UserInfoB, terminal=True), 'experiment')
    
    
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
    UserInfoB['simulation'].slicingDim = [0]
    UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
    predict_Thalamus_For_SD0(UserInfoB)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30
    UserInfoB['simulation'].slicingDim = [1]
    UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
    main(UserInfoB)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].slicingDim = [2]
    UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
    main(UserInfoB)    

    
    merge_results_and_apply_25D(UserInfoB)

def apply():
    UserInfo = terminalEnteries(UserInfo)

    print(UserInfo['Input'].directory)
    # data = nib.load(UserInfo['Input'].directory)




# apply()
