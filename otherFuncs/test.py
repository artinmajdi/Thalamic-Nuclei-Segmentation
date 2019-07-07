import keras.models as kmodel
import sys, os
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import keras.layers as KLayers
from tqdm import tqdm
import numpy as np

dir = '/array/ssd/msmajdi/experiments/keras/exp6/models/sE12_Cascade_FM20_U-Net4_NL3_LS_MyBCE_US1_Main_Init_3T_CV_a/MultiClass_24567891011121314/sd2/'


a = [1,2,3]
b= [5,4,3]


params = paramFunc.Run(UserInfo.__dict__, terminal=False)
ModelParam = params.WhichExperiment.HardParams.Model

if ModelParam.Upsample.Mode:
    scale , szI = ModelParam.Upsample.Scale   ,  [228,288,168]        
    InDim_new = (scale*szI[0] , scale*szI[1] , szI[2] , 1)
    input_shape = tuple(InDim_new[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
else:
    input_shape = tuple(ModelParam.InputDimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)


TF = ModelParam.Transfer_Learning
LP = ModelParam.Layer_Params        
KN = ModelParam.Layer_Params.ConvLayer.Kernel_size        
AC = ModelParam.Layer_Params.Activitation
DT = ModelParam.Layer_Params.Dropout
FM_FCN = ModelParam.Layer_Params.FCN_FeatureMaps

# input_shape = tuple(Input_Dimensions[:ModelParam.Method.InputImage2Dvs3D]) + (1,)
padding      = ModelParam.Layer_Params.ConvLayer.padding        
FCN1_NLayers = ModelParam.FCN1_NLayers
FCN2_NLayers = ModelParam.FCN2_NLayers
num_classes  = ModelParam.MultiClass.num_classes
pool_size    = ModelParam.Layer_Params.MaxPooling.pool_size

def Layer(featureMaps, trainable, input):
    conv = KLayers.Conv2D(featureMaps, kernel_size=KN.conv, padding=padding, trainable=trainable)(input)
    conv = KLayers.BatchNormalization()(conv)  
    return KLayers.Activation(AC.layers)(conv) 

def best_Unet(Unet_input):  
    FM      = ModelParam.Layer_Params.FirstLayer_FeatureMap_Num
    NLayers = ModelParam.num_Layers    

    def Unet_sublayer_Contracting(inputs):
        def main_USC(WBp, nL):
            trainable = TF.U_Net4.Contracting[nL] if TF.Mode else True
            featureMaps = FM*(2**nL)

            conv = Layer(featureMaps, trainable, WBp)
            conv = Layer(featureMaps, trainable, conv)                                              
            
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
            
            if DT.Mode and trainable: conv = KLayers.Dropout(DT.Value)(conv)
            return conv

        for nL in reversed(range(NLayers -1)):  
            WB = main_USE(WB, nL, Conv_Out)

        return WB

    def Unet_MiddleLayer(WB, nL):
        trainable = TF.U_Net4.Middle if TF.Mode else True
        featureMaps = FM*(2**nL)

        WB = Layer(featureMaps, trainable, WB)
        WB = Layer(featureMaps, trainable, WB)   

        if DT.Mode and trainable: WB = KLayers.Dropout(DT.Value)(WB)
        return WB
                            
    WB, Conv_Out = Unet_sublayer_Contracting(Unet_input)
    WB = Unet_MiddleLayer(WB , NLayers-1)
    WB = Unet_sublayer_Expanding(WB , Conv_Out)

    return WB

def FCN_Layer(conv,num_Layers):
    for _ in range(num_Layers):
        conv = Layer(FM_FCN, True, conv)                     
        conv = KLayers.Dropout(DT.Value)(conv)  
    return conv

inputs = KLayers.Input(input_shape)
FCN1   = FCN_Layer(inputs, FCN1_NLayers)    # num layers: nL * 4 + 1
Unet1  = best_Unet(FCN1)
FCN2   = FCN_Layer(Unet1, FCN2_NLayers) 

output = KLayers.Conv2D(num_classes, kernel_size=KN.output, padding=padding, activation=AC.output)(FCN2)
modelNew = kmodel.Model(inputs=[inputs], outputs=[output])

best_WMn_model = kmodel.load_model(dir + 'model.h5') # num_Layers 43
for l in tqdm(range(2,len(best_WMn_model.layers)-1)):
    modelNew.layers[l+FCN1_NLayers*4].set_weights(best_WMn_model.layers[l].get_weights())

print('finished')

