import os, sys
sys.path.append(os.path.dirname(__file__))
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess
from shutil import copyfile , copytree
import tensorflow as tf
from keras import backend as K
import pandas as pd
import xlsxwriter
import csv
import json
class InitValues:
    def __init__(self, Nuclei_Indexes=1 , slicingDim=2):
        self.slicingDim     = slicingDim.copy()

        if Nuclei_Indexes == 'all':  
             _, self.Nuclei_Indexes,_ = smallFuncs.NucleiSelection(ind = 1)
        else:
            self.Nuclei_Indexes = Nuclei_Indexes.copy()
                
def Run(UserInfoB,InitValues):

    def HierarchicalStages(UserInfoB):

        BB = smallFuncs.NucleiIndex(1,'HCascade')
        
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

    def CacadeStages(UserInfoB):

        for UserInfoB['simulation'].nucleus_Index in InitValues.Nuclei_Indexes:
            Run_SingleNuclei(UserInfoB)

    def Run_SingleNuclei(UserInfoB):

        for sd in InitValues.slicingDim:
            
            if not(sd == 0 and UserInfoB['simulation'].nucleus_Index) == 1:

                UserInfoB['simulation'].slicingDim = [sd]                       
                params = paramFunc.Run(UserInfoB, terminal=False)

                print('---------------------------------------------------------------')
                print(' Nucleus:', UserInfoB['simulation'].nucleus_Index  , ' | GPU:', UserInfoB['simulation'].GPU_Index , ' | slicingDim',sd, \
                    ' | Dropout', UserInfoB['DropoutValue'] , ' | Learning_Rate' , UserInfoB['simulation'].Learning_Rate, ' | num_Layers' , UserInfoB['simulation'].num_Layers,\
                    ' | MultThalmaus',UserInfoB['simulation'].Multiply_By_Thalmaus , ' | ', UserInfoB['Model_Method'])

                print('SubExperiment:', params.WhichExperiment.SubExperiment.name)
                print('---------------------------------------------------------------')
                                        
                Data, params = datasets.loadDataset(params)                
                choosingModel.check_Run(params, Data)              
                # K.clear_session()
                 
    if   UserInfoB['Model_Method'] == 'HCascade':  HierarchicalStages(UserInfoB)
    elif UserInfoB['Model_Method'] == 'Cascade' :  CacadeStages(UserInfoB)
    elif UserInfoB['Model_Method'] == 'singleRun': Run_SingleNuclei(UserInfoB)

def preMode(UserInfoB):
    # UserInfoB = smallFuncs.terminalEntries(UserInfoB)
    params = paramFunc.Run(UserInfoB, terminal=True)   
    datasets.movingFromDatasetToExperiments(params)
    applyPreprocess.main(params, 'experiment')
    K = smallFuncs.gpuSetting(params.WhichExperiment.HardParams.Machine.GPU_Index)
    return UserInfoB, K
UserInfoB, K = preMode(UserInfo.__dict__)
UserInfoB['simulation'].verbose = 2


IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

# UserInfoB['Model_Method'] =  'Cascade'
# print('slicingDim' , IV.slicingDim , 'Nuclei_Indexes' , IV.Nuclei_Indexes , 'GPU:  ', UserInfoB['simulation'].GPU_Index, UserInfoB['Model_Method'])
# Run(UserInfoB, IV)


# UserInfoB['Model_Method'] =  'HCascade'
print('slicingDim' , IV.slicingDim , 'Nuclei_Indexes' , IV.Nuclei_Indexes , 'GPU:  ', UserInfoB['simulation'].GPU_Index, UserInfoB['Model_Method'])
Run(UserInfoB, IV)


K.clear_session()
