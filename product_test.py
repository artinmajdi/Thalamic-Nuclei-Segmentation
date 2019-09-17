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
# import json
import nibabel as nib
# from shutil import copyfile , copytree
from tqdm import tqdm
from preprocess import BashCallingFunctionsA, croppingA

class InitValues:
    def __init__(self, Nuclei_Indexes=1 , slicingDim=2):
        self.slicingDim     = slicingDim 
        self.Nuclei_Indexes = smallFuncs.Nuclei_Class(1,'Cascade').All_Nuclei().Indexes if Nuclei_Indexes == 'all' else Nuclei_Indexes
 

def main(UserInfoB):
        
    def func_predict(UserInfoB):
        if not ( (0 in UserInfoB['simulation'].slicingDim) and (1 in UserInfoB['simulation'].nucleus_Index)  ):
            params = paramFunc.Run(UserInfoB, terminal=False)
            Data, params = datasets.loadDataset(params)                             
            choosingModel.check_Run(params, Data)              
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

