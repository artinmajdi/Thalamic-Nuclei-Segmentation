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

        print('************ stage 1 ************')
        UserInfoB['simulation'].nucleus_Index = [1]
        if not check_if_num_Layers_fit(UserInfoB):
            Run_Main(UserInfoB)

            print('************ stage 2 ************')                    
            UserInfoB['simulation'].nucleus_Index = [1.1, 1.2, 1.3, 2]
            if not check_if_num_Layers_fit(UserInfoB):
                Run_Main(UserInfoB)

                print('************ stage 3 ************')
                if not UserInfoB['temp_just_superGroups']:
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
        if not UserInfoB['simulation'].Multi_Class_Mode: NI = [NI]

        
        def subRun(UserInfoB): 
            
            def func_copy_Thalamus_preds(params):                

                def func_mainCopy(name_Thalmus_network):
                    # input_model  = params.WhichExperiment.Experiment.address + '/models/' + name_Thalmus_network # params.WhichExperiment.SubExperiment.name_Thalmus_network
                    # output_model = params.WhichExperiment.Experiment.address + '/models/' + params.WhichExperiment.SubExperiment.name
                    # os.system('mkdir %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

                    input_model  = params.WhichExperiment.Experiment.address + '/results/' + name_Thalmus_network # params.WhichExperiment.SubExperiment.name_Thalmus_network
                    output_model = params.WhichExperiment.Experiment.address + '/results/' + params.WhichExperiment.SubExperiment.name
                    os.system('mkdir %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

                ReadTrain = params.WhichExperiment.Dataset.ReadTrain                
                if ReadTrain.SRI:    func_mainCopy('Predictions_Full_THALAMUS/SRI') 
                if ReadTrain.ET:     func_mainCopy('Predictions_Full_THALAMUS/ET') 
                if ReadTrain.Main:   func_mainCopy('Predictions_Full_THALAMUS/Main') 
                if ReadTrain.CSFn1:  func_mainCopy('Predictions_Full_THALAMUS/CSFn1') 
                if ReadTrain.CSFn2:  func_mainCopy('Predictions_Full_THALAMUS/CSFn2') 
                
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
            Flag_3T = ('sE8' in params.WhichExperiment.SubExperiment.name)

            print_func(UserInfoB, params)
            Read = params.WhichExperiment.Dataset.ReadTrain
            if (1 in NI ) and (UserInfoB['CrossVal'].index == ['a']) and (not Flag_3T): func_copy_Thalamus_preds(params)
            # elif (1.2 in NI) and UserInfoB['simulation'].Multi_Class_Mode and (not Read.ET): print('skipped')
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

def EXP_3_SRI_Main_US2_m2_(UserInfoB):
    UserInfoB['upsample'].Scale = 2
    UserInfoB['simulation'].num_Layers = 5
    UserInfoB['Model_Method'] = 'HCascade'
    UserInfoB['simulation'].batch_size = 30
    # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 50 # , 50
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['TypeExperiment'] in [1, 2]:
        Run(UserInfoB, IV)

def EXP_2_ET_superGroups_Only_HCascade_finetune(UserInfoB):
    
    UserInfoB['simulation'].GPU_Index = "1"
    UserInfoB['Model_Method'] = 'HCascade' # , 'HCascade']:
    UserInfoB['upsample'].Scale = 1
    UserInfoB['TypeExperiment'] = 4
    UserInfoB['temp_just_superGroups'] = True
    UserInfoB['simulation'].nucleus_Index = [1.1, 1.2, 1.3, 1.4]
    UserInfoB['simulation'].batch_size = 100
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['simulation'].num_Layers in [3, 4]:
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 ,30 ,40]: 
            Run(UserInfoB, IV)

def EXP_2b_ET_Cascade_finetune(UserInfoB):
    
    UserInfoB['simulation'].GPU_Index = "4"
    UserInfoB['Model_Method'] = 'Cascade' # , 'HCascade']:
    UserInfoB['upsample'].Scale = 1
    UserInfoB['TypeExperiment'] = 4
    UserInfoB['simulation'].batch_size = 200
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['simulation'].num_Layers in [3, 4]:
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 ,30 ,40]: 
            Run(UserInfoB, IV)

def EXP_1_FM10_allMethods_HCascade(UserInfoB):
    UserInfoB['Model_Method'] = 'HCascade' # , 'HCascade']:
    UserInfoB['upsample'].Scale = 1
    UserInfoB['simulation'].num_Layers = 4
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 10
    UserInfoB['simulation'].batch_size = 100
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['TypeExperiment'] in [1 ,2 ,4]: 
        Run(UserInfoB, IV)

def EXP4_FCN_Unet(UserInfoB):
    # TODO I need to remoev ET ones from this experiment and repeat it, because they weren't initialized
    UserInfoB['Model_Method'] = 'Cascade' # , 'HCascade']:
    UserInfoB['upsample'].Scale = 1
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].batch_size = 100
    for UserInfoB['simulation'].FCN_FeatureMaps in [10 ,20 , 60]: 
        for UserInfoB['TypeExperiment'] in [1 ,2 ,4]: 
            Run(UserInfoB, IV)

def EXP5_Resnet_Cascade_3T_and_Main(UserInfoB):
    UserInfoB['simulation'].GPU_Index = "6"

    # Cascade   Main Init 3T
    UserInfoB['Model_Method'] = 'Cascade'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 10
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [0]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['architectureType'] in ['Res_Unet' , 'SegNet_Unet']: # 'Res_Unet' , 'FCN_Unet', 'SegNet_Unet']:
        for UserInfoB['TypeExperiment'] in [1,2]:
            Run(UserInfoB, IV)

    UserInfoB['Model_Method'] = 'Cascade'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].num_Layers = 4
    UserInfoB['simulation'].slicingDim = [1]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['architectureType'] in ['Res_Unet' , 'SegNet_Unet']: # 'Res_Unet' , 'FCN_Unet', 'SegNet_Unet']:
        for UserInfoB['TypeExperiment'] in [1,2]:
            Run(UserInfoB, IV)

    UserInfoB['Model_Method'] = 'Cascade'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [2]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['architectureType'] in ['Res_Unet' , 'SegNet_Unet']: # 'Res_Unet' , 'FCN_Unet', 'SegNet_Unet']:
        for UserInfoB['TypeExperiment'] in [1,2]:
            Run(UserInfoB, IV)

def EXP6_Resnet_HCascade_3T_and_Main(UserInfoB):

    UserInfoB['simulation'].GPU_Index = "5"

    # ! HCascade   Main Init 3T
    UserInfoB['Model_Method'] = 'HCascade'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [0]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['architectureType'] in ['Res_Unet' , 'SegNet_Unet']: # 'Res_Unet' , 'FCN_Unet', 'SegNet_Unet']:
        for UserInfoB['TypeExperiment'] in [1,2]:
            Run(UserInfoB, IV)

    UserInfoB['Model_Method'] = 'HCascade'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
    UserInfoB['simulation'].num_Layers = 4
    UserInfoB['simulation'].slicingDim = [1]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['architectureType'] in ['Res_Unet' , 'SegNet_Unet']: # 'Res_Unet' , 'FCN_Unet', 'SegNet_Unet']:
        for UserInfoB['TypeExperiment'] in [1,2]:
            Run(UserInfoB, IV)

    UserInfoB['Model_Method'] = 'HCascade'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [2]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['architectureType'] in ['Res_Unet' , 'SegNet_Unet']: # 'Res_Unet' , 'FCN_Unet', 'SegNet_Unet']:
        for UserInfoB['TypeExperiment'] in [1,2]:
            Run(UserInfoB, IV)

def EXP7_Resnet_Cascade_3T_Main_ET(UserInfoB):

    UserInfoB['simulation'].GPU_Index = "3"

    # Cascade   ET Init Main
    UserInfoB['Model_Method'] = 'Cascade'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [0,1]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['architectureType'] in ['Res_Unet' , 'SegNet_Unet']: # 'Res_Unet' , 'FCN_Unet', 'SegNet_Unet']:
        for UserInfoB['TypeExperiment'] in [1,2,4]:
            Run(UserInfoB, IV)

def EXP8_Resnet_HCascade_3T_Main_ET(UserInfoB):

    UserInfoB['simulation'].GPU_Index = "1"

    # HCascade   ET Init Main
    UserInfoB['Model_Method'] = 'HCascade'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [2,1,0]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['architectureType'] in ['Res_Unet' ,'SegNet_Unet']: #  'FCN_Unet', 'Res_Unet' , 'FCN_Unet', 'SegNet_Unet']:
        for UserInfoB['TypeExperiment'] in [1,2,4]:
            Run(UserInfoB, IV)

def EXP9_ET_HCascade(UserInfoB):
    UserInfoB['simulation'].GPU_Index = "0"

    # HCascade   ET Init Main
    UserInfoB['Model_Method'] = 'HCascade'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [2,1,0]
    UserInfoB['tag_temp'] = '_NEW' 
    UserInfoB['Experiments'].Index = '7'
    UserInfoB['architectureType'] = 'U-Net4'
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['TypeExperiment'] in [4,5]:
        Run(UserInfoB, IV)

def EXP10_Unet_Cascade_Main_OtherFolds(UserInfoB):
    # Cascade   Main Init 3T

    # UserInfoB['simulation'].GPU_Index = "5"
    UserInfoB['CrossVal'].index   = ['b']
    UserInfoB['TypeExperiment']   = 2
    UserInfoB['architectureType'] = 'U-Net4'
    UserInfoB['Model_Method']     = 'Cascade'
    UserInfoB['Experiments'].Index = '6'

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 10
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [0]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    Run(UserInfoB, IV)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].num_Layers = 4
    UserInfoB['simulation'].slicingDim = [1]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    Run(UserInfoB, IV)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [2]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    Run(UserInfoB, IV)

def EXP11_Unet_HCascade_Main_OtherFolds(UserInfoB):

    # UserInfoB['simulation'].GPU_Index = "5"
    UserInfoB['CrossVal'].index   = ['b']
    UserInfoB['TypeExperiment']   = 2
    UserInfoB['architectureType'] = 'U-Net4'
    UserInfoB['Model_Method']     = 'HCascade'
    UserInfoB['Experiments'].Index = '6'


    # ! HCascade   Main Init 3T
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [2,0]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    Run(UserInfoB, IV)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
    UserInfoB['simulation'].num_Layers = 4
    UserInfoB['simulation'].slicingDim = [1]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    Run(UserInfoB, IV)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['simulation'].slicingDim = [2]
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    Run(UserInfoB, IV)

def EXP12_SingleClass(UserInfoB):
    UserInfoB['simulation'].Multi_Class_Mode = False    
    UserInfoB['architectureType'] = 'U-Net4'
    UserInfoB['Experiments'].Index = '6'
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].num_Layers = 3
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)


    for UserInfoB['Model_Method'] in ['Cascade' , 'HCascade' , 'mUnet']: 
        for UserInfoB['TypeExperiment'] in [1, 2, 4, 8]: 
            Run(UserInfoB, IV)

def EXP_13_CSFn2_Cascade_finetune(UserInfoB):
    
    # UserInfoB['simulation'].GPU_Index = "0"
    UserInfoB['Model_Method'] = 'Cascade' # , 'HCascade']:
    UserInfoB['upsample'].Scale = 1
    UserInfoB['TypeExperiment'] = 8
    UserInfoB['simulation'].batch_size = 100
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

    for UserInfoB['simulation'].num_Layers in [3, 4]:
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 ,30 ,40]: 
            
            Run(UserInfoB, IV)

UserInfoB, K = preMode(UserInfo.__dict__)

EXP10_Unet_Cascade_Main_OtherFolds(UserInfoB)
EXP11_Unet_HCascade_Main_OtherFolds(UserInfoB)

K.clear_session()