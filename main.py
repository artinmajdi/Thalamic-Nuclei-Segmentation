import os
import sys
sys.path.append(os.path.dirname(__file__))  # sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess
import tensorflow as tf
from keras import backend as K
import pandas as pd
import xlsxwriter
import csv
import json
import nibabel as nib
# from shutil import copyfile , copytree

class InitValues:
    def __init__(self, Nuclei_Indexes=1 , slicingDim=2):
        self.slicingDim     = slicingDim 
        self.Nuclei_Indexes = smallFuncs.Nuclei_Class(1,'Cascade').All_Nuclei().Indexes if Nuclei_Indexes == 'all' else Nuclei_Indexes
                
def Run(UserInfoB,InitValues):

    def print_FullExp(IV):
        print('---------------------------------------------------------------')
        print('Full Experiment Info')
        print('slicingDim' , IV.slicingDim , 'Nuclei_Indexes' , IV.Nuclei_Indexes , 'GPU:  ', UserInfoB['simulation'].GPU_Index, UserInfoB['Model_Method'] , '\n')
        print('---------------------------------------------------------------')
    print_FullExp(InitValues)

    def HierarchicalStages(UserInfoB):

        BB = smallFuncs.Nuclei_Class(1,'HCascade')

        print('************ stage 1 ************')
        if 1 in InitValues.Nuclei_Indexes: 
            UserInfoB['simulation'].nucleus_Index = 1
            Run_SingleNuclei(UserInfoB)

        print('************ stage 2 ************')                    
        for UserInfoB['simulation'].nucleus_Index in BB.HCascade_Parents_Identifier(InitValues.Nuclei_Indexes):
            Run_SingleNuclei(UserInfoB)

        print('************ stage 3 ************')
        for UserInfoB['simulation'].nucleus_Index in BB.remove_Thalamus_From_List(InitValues.Nuclei_Indexes):
            Run_SingleNuclei(UserInfoB)

    def Loop_All_Nuclei(UserInfoB):

        for UserInfoB['simulation'].nucleus_Index in InitValues.Nuclei_Indexes:
            Run_SingleNuclei(UserInfoB)

    def Run_SingleNuclei(UserInfoB):
        def subRun(UserInfoB):             
            params = paramFunc.Run(UserInfoB, terminal=False)

            print('---------------------------------------------------------------')
            print(' Nucleus:', UserInfoB['simulation'].nucleus_Index  , ' | GPU:', UserInfoB['simulation'].GPU_Index , ' | SD',sd, \
                ' | Dropout', UserInfoB['DropoutValue'] , ' | LR' , UserInfoB['simulation'].Learning_Rate, ' | NL' , UserInfoB['simulation'].num_Layers,\
                ' | ', UserInfoB['Model_Method'] , '|  FM', UserInfoB['simulation'].FirstLayer_FeatureMap_Num)

            print('Experiment:', params.WhichExperiment.Experiment.name)                              
            print('SubExperiment:', params.WhichExperiment.SubExperiment.name.replace('_','-'))
            print('---------------------------------------------------------------')
                                    
            Data, params = datasets.loadDataset(params) 
            print(Data.Train.Image.shape)               
            choosingModel.check_Run(params, Data)              
            K.clear_session()

        for sd in InitValues.slicingDim:            
            if not (sd == 0 and UserInfoB['simulation'].nucleus_Index == 1):
                UserInfoB['simulation'].slicingDim = [sd]       

                if UserInfoB['CrossVal'].Mode:
                    for UserInfoB['CrossVal'].index in UserInfoB['CrossVal'].All_Indexes:   
                        subRun(UserInfoB)
                else:
                    subRun(UserInfoB)
                 
    if   UserInfoB['Model_Method'] == 'HCascade':  HierarchicalStages(UserInfoB)
    elif UserInfoB['Model_Method'] == 'Cascade' :  Loop_All_Nuclei(UserInfoB)
    elif UserInfoB['Model_Method'] == 'FCN_25D':   Loop_All_Nuclei(UserInfoB)
    elif UserInfoB['Model_Method'] == 'singleRun': Run_SingleNuclei(UserInfoB)
     
def preMode(UserInfoB):
    UserInfoB = smallFuncs.terminalEntries(UserInfoB)
    params = paramFunc.Run(UserInfoB, terminal=False)   
    datasets.movingFromDatasetToExperiments(params)
    applyPreprocess.main(params, 'experiment')
    K = smallFuncs.gpuSetting(params.WhichExperiment.HardParams.Machine.GPU_Index)
    return UserInfoB, K

UserInfoB, K = preMode(UserInfo.__dict__)

IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

for UserInfoB['Model_Method'] in ['HCascade'  ,  'Cascade' ]:
    Run(UserInfoB, IV)

K.clear_session()
