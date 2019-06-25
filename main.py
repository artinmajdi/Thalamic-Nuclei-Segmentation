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
                
def Run(UserInfoB, InitValues):
    
    
    MM = UserInfoB['Model_Method']

    def check_if_num_Layers_fit(UserInfoB):
        def print_func2(UserInfoB, temp_params):
            print('---------------------- check Layers Step ------------------------------')
            print(' N:', UserInfoB['simulation'].nucleus_Index  , ' | GPU:', UserInfoB['simulation'].GPU_Index , ' | SD',UserInfoB['simulation'].slicingDim[0], \
                ' | Dropout', UserInfoB['DropoutValue'] , ' | LR' , UserInfoB['simulation'].Learning_Rate, ' | NL' , UserInfoB['simulation'].num_Layers,\
                ' | ', UserInfoB['Model_Method'] , '|  FM', UserInfoB['simulation'].FirstLayer_FeatureMap_Num  ,  '|  Upsample' , UserInfoB['upsample'].Scale)

            print('\n' ,'#layer',temp_params.WhichExperiment.HardParams.Model.num_Layers , '#layers changed' , temp_params.WhichExperiment.HardParams.Model.num_Layers_changed)
            # print('---------------------------------------------------------------')                

        temp_params = paramFunc.Run(UserInfoB, terminal=False)
        temp_params = preAnalysis(temp_params)
        print_func2(UserInfoB, temp_params)
        return temp_params.WhichExperiment.HardParams.Model.num_Layers_changed
        
    def HierarchicalStages_single_Class(UserInfoB):

        BB = smallFuncs.Nuclei_Class(1,'HCascade')

        print('************ stage 1 ************')
        if 1 in InitValues.Nuclei_Indexes: 
            UserInfoB['simulation'].nucleus_Index = [1]
            if not check_if_num_Layers_fit(UserInfoB): Run_Main(UserInfoB)

        print('************ stage 2 ************')                    
        for UserInfoB['simulation'].nucleus_Index in BB.HCascade_Parents_Identifier(InitValues.Nuclei_Indexes):
            if not check_if_num_Layers_fit(UserInfoB): Run_Main(UserInfoB)

        print('************ stage 3 ************')
        for UserInfoB['simulation'].nucleus_Index in BB.remove_Thalamus_From_List(InitValues.Nuclei_Indexes):
            if not check_if_num_Layers_fit(UserInfoB): Run_Main(UserInfoB)

    def HierarchicalStages_Multi_Class(UserInfoB):

        # BB = smallFuncs.Nuclei_Class(1,'HCascade')

        print('************ stage 1 ************')
        # if 1 in InitValues.Nuclei_Indexes: 
        UserInfoB['simulation'].nucleus_Index = [1]
        if not check_if_num_Layers_fit(UserInfoB):
            Run_Main(UserInfoB)

            print('************ stage 2 ************')                    
            UserInfoB['simulation'].nucleus_Index = [1.1, 1.2, 1.3, 2]
            if not check_if_num_Layers_fit(UserInfoB):
                Run_Main(UserInfoB)

                print('************ stage 3 ************')
                for parent in [1.1, 1.2, 1.3]:
                    CC = smallFuncs.Nuclei_Class(parent,'HCascade')
                    UserInfoB['simulation'].nucleus_Index = CC.child
                    if not check_if_num_Layers_fit(UserInfoB):
                        Run_Main(UserInfoB)

    def Loop_Over_Nuclei(UserInfoB):

        if not UserInfoB['simulation'].Multi_Class_Mode:
            for UserInfoB['simulation'].nucleus_Index in InitValues.Nuclei_Indexes:
                if not check_if_num_Layers_fit(UserInfoB): Run_Main(UserInfoB)

        else:
                      
            UserInfoB['simulation'].nucleus_Index = [1]
            if not check_if_num_Layers_fit(UserInfoB): 
                Run_Main(UserInfoB)

                BB = smallFuncs.Nuclei_Class(1,'Cascade')            
                UserInfoB['simulation'].nucleus_Index = BB.remove_Thalamus_From_List(list(BB.All_Nuclei().Indexes))
                if not check_if_num_Layers_fit(UserInfoB): Run_Main(UserInfoB)

    def Run_Main(UserInfoB):

        NI = UserInfoB['simulation'].nucleus_Index

        def subRun(UserInfoB): 
            
            def func_copy_Thalamus_preds(params):                

                def func_mainCopy(name_Thalmus_network):
                    input_model  = params.WhichExperiment.Experiment.address + '/models/' + name_Thalmus_network # params.WhichExperiment.SubExperiment.name_Thalmus_network
                    output_model = params.WhichExperiment.Experiment.address + '/models/' + params.WhichExperiment.SubExperiment.name
                    os.system('mkdir %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

                    input_model  = params.WhichExperiment.Experiment.address + '/results/' + name_Thalmus_network # params.WhichExperiment.SubExperiment.name_Thalmus_network
                    output_model = params.WhichExperiment.Experiment.address + '/results/' + params.WhichExperiment.SubExperiment.name
                    os.system('mkdir %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

                ReadTrain = params.WhichExperiment.Dataset.ReadTrain                
                if ReadTrain.SRI:    func_mainCopy('sE8_Predictions_Full_THALAMUS') 
                if ReadTrain.ET:     func_mainCopy('sE12_Predictions_Full_THALAMUS_ET') 
                if ReadTrain.Main:   func_mainCopy('sE12_Predictions_Full_THALAMUS_Main') 
                if ReadTrain.CSFn1:  func_mainCopy('sE12_Predictions_Full_THALAMUS_CSFn1') 
                if ReadTrain.CSFn2:  func_mainCopy('sE12_Predictions_Full_THALAMUS_CSFn2') 
                
            def print_func(UserInfoB, params):
                print('---------------------------------------------------------------')
                print(' Nucleus:', NI  , ' | GPU:', UserInfoB['simulation'].GPU_Index , ' | SD',UserInfoB['simulation'].slicingDim[0], \
                    ' | Dropout', UserInfoB['DropoutValue'] , ' | LR' , UserInfoB['simulation'].Learning_Rate, ' | NL' , UserInfoB['simulation'].num_Layers,\
                    ' | ', UserInfoB['Model_Method'] , '|  FM', UserInfoB['simulation'].FirstLayer_FeatureMap_Num  ,  '|  Upsample' , UserInfoB['upsample'].Scale)

                print('Experiment:', params.WhichExperiment.Experiment.name)                              
                print('SubExperiment:', params.WhichExperiment.SubExperiment.name)
                print('---------------------------------------------------------------')
                            
            def save_Anteior_BBox(params):
                def cropBoundingBoxes(mode, subject):

                    msk = nib.load(subject.Temp.address + '/CropMask_AV.nii.gz')

                    BB = smallFuncs.findBoundingBox(msk.get_data())
                    BBd = [  [BB[ii][0] , BB[ii][1]] for ii in range(len(BB))]
                    
                    dirr = params.directories.Test.Result
                    if 'train' in mode: dirr += '/TrainData_Output'
                    
                    smallFuncs.mkDir(dirr + '/' + subject.subjectName)
                    np.savetxt(dirr + '/' + subject.subjectName + '/BB_' + params.WhichExperiment.Nucleus.name + '.txt',np.concatenate((BB,BBd),axis=1),fmt='%d')

                def loop_Subjects(mode):
                    Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects
                    for _, subject in tqdm(Subjects.items(), desc='Saving Anterior BoundingBox ' + mode):
                        print(subject.subjectName)
                        BashCallingFunctionsA.RigidRegistration_2AV( subject , params.WhichExperiment.HardParams.Template , params.preprocess)
                        croppingA.crop_AV(subject , params)

                        cropBoundingBoxes(mode, subject)
                
                loop_Subjects('train')
                loop_Subjects('test') 
                            
            def normal_run(params):
                """
                def copy_model_for_Transfer_Learning():
                    def func_mainCopy(name_Thalmus_network):
                        input_model  = params.WhichExperiment.Experiment.address + '/models/' + name_Thalmus_network # params.WhichExperiment.SubExperiment.name_Thalmus_network
                        output_model = params.WhichExperiment.Experiment.address + '/models/' + params.WhichExperiment.SubExperiment.name
                        os.system('mkdir %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

                        input_model  = params.WhichExperiment.Experiment.address + '/results/' + name_Thalmus_network # params.WhichExperiment.SubExperiment.name_Thalmus_network
                        output_model = params.WhichExperiment.Experiment.address + '/results/' + params.WhichExperiment.SubExperiment.name
                        os.system('mkdir %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

                    # params.

                    # Model_3T       = Exp_address + '/models/' + SE.name_Init_from_3T      + '/' + NucleusName  + sdTag
                    # Model_7T       = Exp_address + '/models/' + SE.name_Init_from_7T      + '/' + NucleusName  + sdTag
                    # Model_CSFn1    = Exp_address + '/models/' + SE.name_Init_from_CSFn1   + '/' + NucleusName  + sdTag

                    if params.UserInfoB['InitializeB'].From_3T:    func_mainCopy('sE8_Predictions_Full_THALAMUS') 
                    if params.UserInfoB['InitializeB'].From_7T:    func_mainCopy('sE8_Predictions_Full_THALAMUS') 
                    if params.UserInfoB['InitializeB'].From_CSFn1: func_mainCopy('sE8_Predictions_Full_THALAMUS') 
                                            
                if params.UserInfoB['Transfer_Learning'].Mode: copy_model_for_Transfer_Learning(params)
                """

                Data, params = datasets.loadDataset(params)                             
                choosingModel.check_Run(params, Data)              
                K.clear_session()
                                
            params = paramFunc.Run(UserInfoB, terminal=False)
            print_func(UserInfoB, params)

            if (NI == [1]): func_copy_Thalamus_preds(params) # print('--')
            elif (NI == [1.4]) and (not UserInfoB['simulation'].Multi_Class_Mode): save_Anteior_BBox(params)
            else: normal_run(params)

        def Loop_slicing_orientations(UserInfoB, InitValues):
            for sd in InitValues.slicingDim:            
                if not (sd == 0 and NI == [1]):                    
                    UserInfoB['simulation'].slicingDim = [sd]
                    subRun(UserInfoB)

        Loop_slicing_orientations(UserInfoB, InitValues)

                 
    if  MM == 'HCascade':  
        if UserInfoB['simulation'].Multi_Class_Mode: HierarchicalStages_Multi_Class(UserInfoB)
        else: HierarchicalStages_single_Class(UserInfoB)
    elif MM == 'singleRun': Run_Main(UserInfoB)
    else: Loop_Over_Nuclei(UserInfoB)
     
def preMode(UserInfoB):
    UserInfoB = smallFuncs.terminalEntries(UserInfoB)
    params = paramFunc.Run(UserInfoB, terminal=False)   
    datasets.movingFromDatasetToExperiments(params)
    applyPreprocess.main(params, 'experiment')
    K = smallFuncs.gpuSetting(params.WhichExperiment.HardParams.Machine.GPU_Index)
    return UserInfoB, K

def Run_tryExcept(UserInfoB, IV):
    try:        
        Run(UserInfoB, IV)
    except Exception as e:
        print('Failed')
        print('Failed')
        print('Failed')
        print(e)
        print( 'US', UserInfoB['upsample'].Scale , 'NL', UserInfoB['simulation'].num_Layers , 'FM', UserInfoB['simulation'].FirstLayer_FeatureMap_Num)
        print('Failed')
        print('Failed')

def loop_fine_tuning(UserInfoB):

    for UserInfoB['upsample'].Scale in [1]: #  2 , 4]:
        for UserInfoB['simulation'].num_Layers in [3 , 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30 , 40 , 60]:
                Run(UserInfoB, IV)

def loop_fine_tuning2(UserInfoB):

    for UserInfoB['upsample'].Scale in [2 , 4]:
        for UserInfoB['simulation'].num_Layers in [5]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30 , 40]:
                Run(UserInfoB, IV)

    for UserInfoB['upsample'].Scale in [4]:
        for UserInfoB['simulation'].num_Layers in [6]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30 , 40]:
                Run(UserInfoB, IV)

def loop_fine_tuning_CSFn(UserInfoB):

    
    # for UserInfoB['simulation'].num_Layers in [3]:
    #     for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [30 , 40 , 60]:
    #         Run(UserInfoB, IV)

    # for UserInfoB['simulation'].num_Layers in [4]:
    #     for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30 , 40 , 60]:
    #         Run(UserInfoB, IV)

    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['upsample'].Scale = 2
    for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 40]:
        Run(UserInfoB, IV)

def func_temp_checkSingleClass_vs_MultiClass(UserInfoB):
    UserInfoB['simulation'].Multi_Class_Mode = False
    for UserInfoB['lossFunction_Index'] in [3, 1, 4]:
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30]:
            Run(UserInfoB, IV)

    UserInfoB['simulation'].Multi_Class_Mode = True
    for UserInfoB['lossFunction_Index'] in [1, 4]:
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30]:
            Run(UserInfoB, IV)

def func_temp2_checkLossFunction(UserInfoB):
    for UserInfoB['lossFunction_Index'] in [4, 6 , 7 , 8 , 9]:
        Run(UserInfoB, IV)


UserInfoB, K = preMode(UserInfo.__dict__)

IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)


# func_temp2_checkLossFunction(UserInfoB)

# func_temp_checkSingleClass_vs_MultiClass(UserInfoB)

for UserInfoB['TypeExperiment'] in [1, 2, 4, 11, 5 , 9 , 10]:
    Run(UserInfoB, IV)

# for UserInfoB['architectureType'] in ['U-Net4', 'FCN_Unet']:
#     for UserInfoB['TypeExperiment'] in [1, 2, 4]:
# Run(UserInfoB, IV)

# loop_fine_tuning(UserInfoB)

# loop_fine_tuning2(UserInfoB)

# loop_fine_tuning_CSFn(UserInfoB)

# Run(UserInfoB, IV)

K.clear_session()
