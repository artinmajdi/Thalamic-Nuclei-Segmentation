import os
import sys
sys.path.append(os.path.dirname(__file__))  # sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append('/code')
print('----------+++',os.path.dirname(__file__))
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
# import tensorflow.compat.v1 as tf2
# tf2.disable_v2_behavior()

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
            UserInfoB['simulation'].nucleus_Index = 1
            if not check_if_num_Layers_fit(UserInfoB): Run_Main(UserInfoB)

        print('************ stage 2 ************')                    
        for UserInfoB['simulation'].nucleus_Index in BB.HCascade_Parents_Identifier(InitValues.Nuclei_Indexes):
            if not check_if_num_Layers_fit(UserInfoB): Run_Main(UserInfoB)

        print('************ stage 3 ************')
        for UserInfoB['simulation'].nucleus_Index in BB.remove_Thalamus_From_List(InitValues.Nuclei_Indexes):
            if not check_if_num_Layers_fit(UserInfoB): Run_Main(UserInfoB)

    def HierarchicalStages_Multi_Class(UserInfoB):

        print('************ stage 1 ************')
        UserInfoB['simulation'].nucleus_Index = 1
        if not check_if_num_Layers_fit(UserInfoB):
            Run_Main(UserInfoB)

            print('************ stage 2 ************')                    
            UserInfoB['simulation'].nucleus_Index = [1.1, 1.2, 1.3, 2]
            if not check_if_num_Layers_fit(UserInfoB):
                Run_Main(UserInfoB)

                print('************ stage 3 ************')
                if not UserInfoB['temp_just_superGroups']:
                    for parent in [ 1.1, 1.2, 1.3]:
                        CC = smallFuncs.Nuclei_Class(parent,'HCascade')
                        UserInfoB['simulation'].nucleus_Index = CC.child
                        if not check_if_num_Layers_fit(UserInfoB):
                            Run_Main(UserInfoB)

    def Loop_Over_Nuclei(UserI):
        
        if not UserI['simulation'].Multi_Class_Mode:
            for UserI['simulation'].nucleus_Index in InitValues.Nuclei_Indexes:
                if not check_if_num_Layers_fit(UserI): Run_Main(UserI)

        else:

            UserI['simulation'].nucleus_Index = [1]
            Flag_Thalmaus_NLayers = check_if_num_Layers_fit(UserI)
            if not Flag_Thalmaus_NLayers: 
                if 1 in InitValues.Nuclei_Indexes: Run_Main(UserI)
            
                BB = smallFuncs.Nuclei_Class(1,'Cascade')            
                A = [InitValues.Nuclei_Indexes] if not isinstance(InitValues.Nuclei_Indexes, list) else InitValues.Nuclei_Indexes
                UserI['simulation'].nucleus_Index = BB.remove_Thalamus_From_List(A) # BB.All_Nuclei().Indexes))
                if UserI['simulation'].nucleus_Index and (not check_if_num_Layers_fit(UserI)): Run_Main(UserI)

    def Run_Main(UserInfoB):

        NI = UserInfoB['simulation'].nucleus_Index
        if not UserInfoB['simulation'].Multi_Class_Mode: NI = [NI]

        def subRun(UserInfoB): 
            
            def func_copy_Thalamus_preds(params):                

                def func_mainCopy(name_Thalmus_network):
                    # input_model  = params.WhichExperiment.Experiment.address + '/models/' + name_Thalmus_network # params.WhichExperiment.SubExperiment.name_Thalmus_network
                    # output_model = params.WhichExperiment.Experiment.address + '/models/' + params.WhichExperiment.SubExperiment.name
                    # os.system('mkdir %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

                    CV = '/CV' + params.WhichExperiment.SubExperiment.crossVal.index[0] if 'SRI' not in name_Thalmus_network else ''                    
                    SD = '/sd' + str(params.WhichExperiment.Dataset.slicingInfo.slicingDim)
                    input_model  = params.WhichExperiment.Experiment.address + '/results/' + name_Thalmus_network + CV + SD # params.WhichExperiment.SubExperiment.name_Thalmus_network
                    output_model = params.WhichExperiment.Experiment.address + '/results/' + params.WhichExperiment.SubExperiment.name + SD
                    os.system('mkdir -p %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

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
            # Flag_CSFn1 = ('CSFn1_Init' in params.WhichExperiment.SubExperiment.name)
            Flag_Unet = (params.WhichExperiment.HardParams.Model.architectureType == 'U-Net4')

            print_func(UserInfoB, params)
            Read = params.WhichExperiment.Dataset.ReadTrain
            if (1 in NI) and UserInfoB['copy_Thalamus']: func_copy_Thalamus_preds(params)  #   and ( (UserInfoB['CrossVal'].index == ['a']) or Flag_3T)     #   and (not Flag_3T)  and (not Flag_CSFn1)
            # elif (1.2 in NI) and UserInfoB['simulation'].Multi_Class_Mode and (Read.ET): print('skipped')
            elif (NI == [1.4]) and (not UserInfoB['simulation'].Multi_Class_Mode): save_Anteior_BBox(params)
            else: normal_run(params)

        def Loop_slicing_orientations(UserInfoB, InitValues):
            for sd in InitValues.slicingDim:            
                if not (sd == 0 and NI == [1]) or UserInfoB['copy_Thalamus']:
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
    # params = paramFunc.Run(UserInfoB, terminal=False)   
    # # datasets.movingFromDatasetToExperiments(params)
    # # applyPreprocess.main(params, 'experiment')
    K = smallFuncs.gpuSetting(str(UserInfoB['simulation'].GPU_Index)) # params.WhichExperiment.HardParams.Machine.GPU_Index)
    return UserInfoB, K

UserInfoB, K = preMode(UserInfo.__dict__)

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


def func_all_experiments(UserInfoB):
    def EXP_3_SRI_Main_US2_m2_(UserInfoB):
        UserInfoB['upsample'].Scale = 2
        UserInfoB['simulation'].num_Layers = 5
        UserInfoB['Model_Method'] = 'HCascade'
        UserInfoB['simulation'].batch_size = 30
        # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 50 # , 50
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['TypeExperiment'] in [1, 2]:
            Run(UserInfoB, IV)

    def EXP_15_ET_HCascade_finetune(UserInfoB):
        
        UserInfoB['Model_Method'] = 'HCascade'
        UserInfoB['upsample'].Scale = 1
        UserInfoB['TypeExperiment'] = 4
        UserInfoB['temp_just_superGroups'] = False
        UserInfoB['simulation'].nucleus_Index = [1.1, 1.2, 1.3, 1.4]
        UserInfoB['simulation'].batch_size = 100
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        UserInfoB['lossFunction_Index'] = 3
        
        for UserInfoB['simulation'].num_Layers in [3, 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 ,30 ,40]: 
                Run(UserInfoB, IV)

    def EXP_15b_ET_HCascade_finetune_JointLoss(UserInfoB):
        
        UserInfoB['Model_Method'] = 'HCascade'
        UserInfoB['upsample'].Scale = 1
        UserInfoB['TypeExperiment'] = 4
        UserInfoB['temp_just_superGroups'] = False
        UserInfoB['simulation'].nucleus_Index = [1.1, 1.2, 1.3, 1.4]
        UserInfoB['simulation'].batch_size = 100
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        UserInfoB['lossFunction_Index'] = 5
        
        for UserInfoB['simulation'].num_Layers in [3, 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 ,30 ,40]: 
                Run(UserInfoB, IV)

    def EXP_2c_ET_nuclei_Only_HCascade_finetune(UserInfoB):
        
        # UserInfoB['simulation'].GPU_Index = "1"
        UserInfoB['Model_Method'] = 'HCascade'
        UserInfoB['upsample'].Scale = 1
        UserInfoB['TypeExperiment'] = 4
        UserInfoB['temp_just_superGroups'] = False
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
        UserInfoB['architectureType'] = 'FCN_Unet'
        UserInfoB['Experiments'].Index = '7'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        
        # for UserInfoB['simulation'].FCN_FeatureMaps in [10 ,20 , 60]:
        #     for UserInfoB['TypeExperiment'] in [4, 5, 8 , 10]: 

        UserInfoB['simulation'].FCN_FeatureMaps = 10
        UserInfoB['TypeExperiment'] = 4
        Run(UserInfoB, IV)

    def EXP5_Resnet_JointDice(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet'
        UserInfoB['lossFunction_Index'] = 5
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)


        for UserInfoB['TypeExperiment'] in [8, 6, 7]: # 1, 2, 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10 , 15 , 20, 30]:        
                Run(UserInfoB, IV)

    def EXP5c_Resnet_BCE_Cascade(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['simulation'].Multi_Class_Mode = True
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet'
        UserInfoB['lossFunction_Index'] = 3
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)


        for UserInfoB['TypeExperiment'] in [1, 2, 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10 , 15 , 20, 30]:        
                Run(UserInfoB, IV)

    def EXP5d_Resnet_JointDice_GeomtericalMean(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet'
        UserInfoB['lossFunction_Index'] = 6
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)


        for UserInfoB['TypeExperiment'] in [1, 2, 4, 8]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10 , 15 , 20, 30]:        
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

        # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        # UserInfoB['simulation'].num_Layers = 4
        # UserInfoB['simulation'].slicingDim = [1]
        # IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        # Run(UserInfoB, IV)

        # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        # UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [2]
        # IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        # Run(UserInfoB, IV)

    def EXP11_Unet_HCascade_Main_OtherFolds(UserInfoB):

        UserInfoB['CrossVal'].index   = ['b']
        UserInfoB['TypeExperiment']   = 2
        UserInfoB['architectureType'] = 'U-Net4'
        UserInfoB['Model_Method']     = 'HCascade'
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['temp_just_superGroups'] = False

        # ! HCascade   Main Init 3T

        # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30
        # UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [0]
        # IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        # Run(UserInfoB, IV)

        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [1]
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        Run(UserInfoB, IV)

        # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
        # UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [2]
        # IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        # Run(UserInfoB, IV)

    def EXP12_SingleClass(UserInfoB):
        UserInfoB['simulation'].Multi_Class_Mode = False    
        UserInfoB['architectureType'] = 'U-Net4'
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        UserInfoB['simulation'].num_Layers = 3
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)


        for UserInfoB['Model_Method'] in ['mUnet']:  # 'HCascade' , 'Cascade' ,
            for UserInfoB['TypeExperiment'] in [1, 2, 4]: 
                Run(UserInfoB, IV)

    def EXP12b_SingleClass(UserInfoB):
        UserInfoB['simulation'].Multi_Class_Mode = False    
        UserInfoB['architectureType'] = 'U-Net4'
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['Model_Method'] = 'Cascade'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        
    
        UserInfoB['TypeExperiment'] = 2
        UserInfoB['CrossVal'].index = ['b']   
        Run(UserInfoB, IV)

        UserInfoB['TypeExperiment'] = 4
        for UserInfoB['CrossVal'].index in ['b' , 'c' , 'd']:       
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

    def EXP_14_CSFn1_Cascade_finetune(UserInfoB):
        
        # UserInfoB['simulation'].GPU_Index = "0"
        UserInfoB['Model_Method'] = 'Cascade' # , 'HCascade']:
        UserInfoB['upsample'].Scale = 1
        UserInfoB['TypeExperiment'] = 6
        UserInfoB['simulation'].batch_size = 50
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['simulation'].num_Layers in [3, 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 ,30 ,40]:             
                Run(UserInfoB, IV)

    def EXP15a_TL_CSFn2(UserInfoB):
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'FCN_Unet_TL'
        UserInfoB['simulation'].FCN1_NLayers = 3
        UserInfoB['simulation'].FCN2_NLayers = 0
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['simulation'].FCN_FeatureMaps in [10, 20 , 30 , 40, 50]:
            # try:
            Run(UserInfoB, IV)
            # except Exception as e:
                # print(e)

    def EXP15b_TL_CSFn2(UserInfoB):
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'FCN_Unet_TL'
        UserInfoB['simulation'].FCN1_NLayers = 3
        UserInfoB['simulation'].FCN2_NLayers = 1
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['simulation'].FCN_FeatureMaps in [10, 20 , 30 , 40, 50]:
            # try:
            Run(UserInfoB, IV)
            # except Exception as e:
            #     print(e)

    def EXP15c1_TL_CSFn2_ResNet_JointLoss(UserInfoB):
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 5
        UserInfoB['Experiments'].Index = '7'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        UserInfoB['simulation'].FCN1_NLayers = 0
        for UserInfoB['simulation'].FCN2_NLayers in [0, 1]:
            for UserInfoB['simulation'].FCN_FeatureMaps in [10, 20 , 30 , 40]:
                Run(UserInfoB, IV)

    def EXP15c2_TL_CSFn2_ResNet_JointLoss(UserInfoB):
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 5
        UserInfoB['Experiments'].Index = '7'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        UserInfoB['simulation'].FCN1_NLayers = 0
        for UserInfoB['simulation'].FCN2_NLayers in [2, 3]:
            for UserInfoB['simulation'].FCN_FeatureMaps in [10, 20 , 30 , 40]:
                Run(UserInfoB, IV)

    def EXP17a_Resnet2_JointDice(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 5
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['TypeExperiment'] in [1, 2, 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10 , 15 , 20 , 30, 40]:        
                Run(UserInfoB, IV)

    def EXP18a_TL_CSFn2_ResNet2_JointLoss(UserInfoB):
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 5
        UserInfoB['Experiments'].Index = '7'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['simulation'].FCN1_NLayers in [0]:
            for UserInfoB['simulation'].FCN2_NLayers in [0, 1, 2]:

                if (UserInfoB['simulation'].FCN1_NLayers == 0) and (UserInfoB['simulation'].FCN2_NLayers == 0):
                    UserInfoB['simulation'].FCN_FeatureMaps = 0
                    Run(UserInfoB, IV)
                else:
                    for UserInfoB['simulation'].FCN_FeatureMaps in [10, 20 , 30 , 40]:
                        Run(UserInfoB, IV)

    def EXP18b2_TL_CSFn2_ResNet2_JointLoss(UserInfoB):
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 5
        UserInfoB['Experiments'].Index = '7'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['simulation'].FCN1_NLayers in [1]:
            for UserInfoB['simulation'].FCN2_NLayers in [2]: # 0, 1,
                for UserInfoB['simulation'].FCN_FeatureMaps in [10, 20 , 30 , 40]:
                    Run(UserInfoB, IV)

        for UserInfoB['simulation'].FCN1_NLayers in [2]:
            for UserInfoB['simulation'].FCN2_NLayers in [2]: # 0, 1, 
                for UserInfoB['simulation'].FCN_FeatureMaps in [20 , 30 , 40]:  # 10, 
                    Run(UserInfoB, IV)

    def EXP18c_TL_CSFn2_ResNet2_JointLoss(UserInfoB):
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 5
        UserInfoB['Experiments'].Index = '7'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['simulation'].FCN1_NLayers in [2]:
            for UserInfoB['simulation'].FCN2_NLayers in [2]: # 0, 1, 

                for UserInfoB['simulation'].FCN_FeatureMaps in [20 , 30 , 40]:  # 10, 
                    Run(UserInfoB, IV)

    def EXP19a_Resnet2_LogDice(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['TypeExperiment'] in [2, 4, 8]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10 , 15]:    
                Run(UserInfoB, IV)

    def EXP19b_Resnet2_LogDice(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['TypeExperiment'] in [2, 4, 8]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30]:
                Run(UserInfoB, IV)

    def EXP19c_Resnet2_LogDice(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['TypeExperiment'] in [2, 4, 8]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [40]:
                Run(UserInfoB, IV)

    def EXP20_Resnet_LogDice(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['TypeExperiment'] in [2, 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10 , 15, 20, 30 , 40]:
                Run(UserInfoB, IV)

    def EXP21_Resnet2_LogDice_InitRn(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['TypeExperiment'] in [14]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10 , 15, 20 , 30, 40]:
                Run(UserInfoB, IV)

    def EXP22_Resnet2_LogDice_LRScheduler(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['simulation'].LR_Scheduler = True
        UserInfoB['Experiments'].Index = '6'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['TypeExperiment'] in [2, 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10 , 15, 20 , 30 , 40]:    
                Run(UserInfoB, IV)

    def EXP24_SingleClass_AV(UserInfoB):
        UserInfoB['simulation'].Multi_Class_Mode = False  
        UserInfoB['simulation'].nucleus_Index = [1,2] 
        # UserInfoB['simulation'].epochs = 30 
        
        # UserInfoB['simulation'].slicingDim = [2] # 2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['temp_copy_sd0'] = True
        # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['Model_Method'] = 'HCascade'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['upsample'].Scale in [1,2]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10, 15, 20, 30, 40]:
                for UserInfoB['TypeExperiment'] in [1, 2, 4]:         
                    Run(UserInfoB, IV)

    def EXP24b_SingleClass_AV(UserInfoB):
        UserInfoB['simulation'].Multi_Class_Mode = False  
        UserInfoB['simulation'].nucleus_Index = [1,2] 
        # UserInfoB['simulation'].epochs = 30 
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['temp_copy_sd0'] = True
        # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['Model_Method'] = 'HCascade'
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['upsample'].Scale in [4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10, 15, 20, 30, 40]:
                for UserInfoB['TypeExperiment'] in [1, 2, 4]:         
                    Run(UserInfoB, IV)


    def EXP25_Res_Unet2_Cascade_Main_OtherFolds(UserInfoB):
        # Cascade   Main Init 3T

        def predict_Thalamus_For_SD0(UserI):

            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)

            # UserI['simulation'].LR_Scheduler = False
            # UserI['TypeExperiment'] = 1
            # Run(UserI, IV)
            
            UserI['simulation'].LR_Scheduler = True
            UserI['TypeExperiment'] = 2
            Run(UserI, IV)


        UserInfoB['CrossVal'].index   = ['b']
        UserInfoB['TypeExperiment']   = 2
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['simulation'].LR_Scheduler = True
        UserInfoB['Experiments'].Index = '7'
        # UserInfoB['temp_copy_sd0'] = False
        UserInfoB['copy_Thalamus'] = False


        # # UserInfoB['simulation'].nucleus_Index = [1]
        # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 15
        # UserInfoB['simulation'].slicingDim = [2]
        # UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14] 
        # UserInfoB['simulation'].num_Layers = 3

        # IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        # Run(UserInfoB, IV)

        # UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        # UserInfoB['simulation'].slicingDim = [1]
        # UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14] 
        # UserInfoB['simulation'].num_Layers = 3
        # IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        # Run(UserInfoB, IV)


        # # sagittal orientation
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        # predict_Thalamus_For_SD0(UserInfoB)


        UserInfoB['simulation'].slicingDim = [0]
        UserInfoB['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14] 
        UserInfoB['simulation'].num_Layers = 3
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        Run(UserInfoB, IV)    

    def EXP25b_Res_Unet2_Cascade_ET_OtherFolds(UserInfoB):
        # Cascade   Main Init 3T
        def main_separateThalamus_sagittal(UserI):
            
            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]               
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

            
            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)
            
        UserInfoB['tag_temp'] = '_LR1e3'
        UserInfoB['copy_Thalamus'] = True
        UserInfoB['TypeExperiment']   = 4
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['simulation'].LR_Scheduler = True
        UserInfoB['Experiments'].Index = '7'

        # UserInfoB['temp_copy_sd0'] = False

        for UserInfoB['CrossVal'].index in ['b', 'c', 'd']:

            UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
            UserInfoB['simulation'].slicingDim = [2]  
            UserInfoB['simulation'].num_Layers = 3
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]           
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)    

            UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
            UserInfoB['simulation'].slicingDim = [1]
            UserInfoB['simulation'].num_Layers = 3
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]    
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV) 

            UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
            UserInfoB['simulation'].slicingDim = [0]
            UserInfoB['simulation'].num_Layers = 3
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV) 
            # main_separateThalamus_sagittal(UserInfoB)

    def EXP25c_TL_CSFn2_ResNet2_DiceLoss_OtherFolds(UserInfoB):
        
        def main_separateThalamus_sagittal(UserI):
            
            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]                
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

            
            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '7'
        UserInfoB['copy_Thalamus'] = False
        UserInfoB['simulation'].LR_Scheduler = False

        for UserInfoB['CrossVal'].index in ['b', 'c', 'd']:
            
            UserInfoB['simulation'].FCN_FeatureMaps = 40
            UserInfoB['simulation'].FCN1_NLayers = 2
            UserInfoB['simulation'].FCN2_NLayers = 0
            UserInfoB['simulation'].slicingDim = [2]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]          
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)    
            Run(UserInfoB, IV)

            UserInfoB['simulation'].FCN_FeatureMaps = 0
            UserInfoB['simulation'].FCN1_NLayers = 0
            UserInfoB['simulation'].FCN2_NLayers = 0
            UserInfoB['simulation'].slicingDim = [1]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]          
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)

            UserInfoB['simulation'].FCN_FeatureMaps = 40
            UserInfoB['simulation'].FCN1_NLayers = 1
            UserInfoB['simulation'].FCN2_NLayers = 1
            UserInfoB['simulation'].slicingDim = [0]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]  
            # IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            # Run(UserInfoB, IV)
            main_separateThalamus_sagittal(UserInfoB)    

    def EXP25c2_TL_CSFn2_ResNet2_DiceLoss_OtherFolds(UserInfoB):
        
        def main_separateThalamus_sagittal(UserI):
            
            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]                
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

            
            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

        UserInfoB['tag_temp'] = '_FrozenLayers2'
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '7'
        UserInfoB['copy_Thalamus'] = False
        UserInfoB['simulation'].LR_Scheduler = False

        for UserInfoB['CrossVal'].index in ['b', 'c', 'd']:
            
            UserInfoB['simulation'].FCN_FeatureMaps = 40
            UserInfoB['simulation'].FCN1_NLayers = 2
            UserInfoB['simulation'].FCN2_NLayers = 0
            UserInfoB['simulation'].slicingDim = [2]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]          
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)    
            Run(UserInfoB, IV)

            UserInfoB['simulation'].FCN_FeatureMaps = 0
            UserInfoB['simulation'].FCN1_NLayers = 0
            UserInfoB['simulation'].FCN2_NLayers = 0
            UserInfoB['simulation'].slicingDim = [1]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]          
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)

            UserInfoB['simulation'].FCN_FeatureMaps = 40
            UserInfoB['simulation'].FCN1_NLayers = 1
            UserInfoB['simulation'].FCN2_NLayers = 1
            UserInfoB['simulation'].slicingDim = [0]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]  
            # IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            # Run(UserInfoB, IV)
            main_separateThalamus_sagittal(UserInfoB)  
            



    def EXP26a_JustThalmaus_3T_Main(UserInfoB):
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '7'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].nucleus_Index = 1
        UserInfoB['copy_Thalamus'] = False
        

        UserInfoB['simulation'].slicingDim = [2] 
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        for  UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [15, 20, 40]:
            UserInfoB['simulation'].LR_Scheduler = False
            UserInfoB['TypeExperiment'] = 1
            Run(UserInfoB, IV)
            
            UserInfoB['simulation'].LR_Scheduler = True
            UserInfoB['TypeExperiment'] = 2
            Run(UserInfoB, IV)
            
    def EXP26b_JustThalmaus_3T_Main(UserInfoB):
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '7'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].nucleus_Index = 1  
        UserInfoB['copy_Thalamus'] = False    

        UserInfoB['simulation'].slicingDim = [1] 
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        for  UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20, 40]:
            UserInfoB['simulation'].LR_Scheduler = False
            UserInfoB['TypeExperiment'] = 1
            Run(UserInfoB, IV)
            
            UserInfoB['simulation'].LR_Scheduler = True
            UserInfoB['TypeExperiment'] = 2
            Run(UserInfoB, IV)


    def EXP27a_Resnet2_LogDice_fineTune_ET_Ps_Main(UserInfoB):
        
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = True
        UserInfoB['TypeExperiment'] = 15
        UserInfoB['simulation'].LR_Scheduler = False    
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)


        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10, 15, 20, 30, 40]:    
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]   
            Run(UserInfoB, IV)


    def EXP27a2_Resnet2_LogDice_fineTune_ET_Ps_Main(UserInfoB):
        
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
        UserInfoB['TypeExperiment'] = 15
        UserInfoB['simulation'].LR_Scheduler = True    
        

        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20,30, 40, 10, 15]: 
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]   
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)

    def EXP27a22_AllTH_Resnet2_LogDice_fineTune_ET_Ps_Main(UserInfoB):
        
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
        UserInfoB['TypeExperiment'] = 15
        
        
        UserInfoB['simulation'].LR_Scheduler = True    
        UserInfoB['simulation'].ReadAugments_Mode = False 


        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10, 15, 20, 30, 40]:   # 
            # if UserInfoB['simulation'].FirstLayer_FeatureMap_Num == 20: UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
            # else: 
            UserInfoB['simulation'].nucleus_Index = [1]
            UserInfoB['simulation'].slicingDim = [2,1,0]
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)



    def EXP27a3_Resnet2_LogDice_fineTune_ET_Ps_Main(UserInfoB):
        
        UserInfoB['Model_Method'] = 'normal'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = True
        UserInfoB['TypeExperiment'] = 15
        UserInfoB['simulation'].LR_Scheduler = False    
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20    
        UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
        # UserInfoB['dataGenerator'].Mode = True
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        Run(UserInfoB, IV)

    def EXP27b_Resnet2_LogDice_fineTune_ET_Ps_Main_Ps_SRI(UserInfoB):
        
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = True
        UserInfoB['TypeExperiment'] = 16
        UserInfoB['simulation'].LR_Scheduler = False    
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)

        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [15, 10, 15, 20, 30, 40]:            
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]   
            Run(UserInfoB, IV)

    def EXP28_Resnet2_LogDice_mUnet_fineTune(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] =  'normal' # 'mUnet'
        UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [2,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['simulation'].LR_Scheduler = False
        UserInfoB['copy_Thalamus'] = True        
        UserInfoB['simulation'].batch_size    = 30
        # for UserInfoB['TypeExperiment'] in [1, 2, 4]:
        #     for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20, 30, 40]:
        #         UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
        #         IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        #         Run(UserInfoB, IV)

        for UserInfoB['TypeExperiment'] in [2, 4]:
            for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20]:
                UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
                UserInfoB['simulation'].slicingDim = [2,0]
                IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
                Run(UserInfoB, IV)

    def EXP29_Resnet2_LogDice_Cascade_InitRn_fineTune(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] =  'Cascade' # 'mUnet'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['simulation'].LR_Scheduler = False
        UserInfoB['copy_Thalamus'] = True
        UserInfoB['TypeExperiment'] = 14
        
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10, 20, 30, 40]:
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
            UserInfoB['simulation'].slicingDim = [2,1]
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)        
            Run(UserInfoB, IV)

    def NOT_Prepared_yet_EXP30_Resnet2_JointLoss_OtherFolds(UserInfoB):
    # Cascade   Main Init 3T

        def predict_Thalamus_For_SD0(UserI):

            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)

            # UserI['simulation'].LR_Scheduler = False
            # UserI['TypeExperiment'] = 1
            # Run(UserI, IV)
            
            UserI['simulation'].LR_Scheduler = True
            UserI['TypeExperiment'] = 2
            Run(UserI, IV)


        UserInfoB['CrossVal'].index   = ['b']
        UserInfoB['TypeExperiment']   = 2
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 5
        UserInfoB['simulation'].LR_Scheduler = False
        UserInfoB['Experiments'].Index = '7'
        UserInfoB['copy_Thalamus'] = False


        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 15
        UserInfoB['simulation'].slicingDim = [2]
        UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14] 
        UserInfoB['simulation'].num_Layers = 3

        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        Run(UserInfoB, IV)

        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        UserInfoB['simulation'].slicingDim = [1]
        UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14] 
        UserInfoB['simulation'].num_Layers = 3
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        Run(UserInfoB, IV)


        # # sagittal orientation
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        # predict_Thalamus_For_SD0(UserInfoB)


        UserInfoB['simulation'].slicingDim = [0]
        UserInfoB['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14] 
        UserInfoB['simulation'].num_Layers = 3
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        Run(UserInfoB, IV)        

    def EXP30_Resnet2_LogDice_fineTune_MainSRI(UserInfoB):
        
        # Cascade   Main Init 3T
        UserInfoB['Model_Method'] =  'Cascade' # 'mUnet'
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['simulation'].LR_Scheduler = False
        UserInfoB['copy_Thalamus'] = True        
        UserInfoB['simulation'].batch_size    = 30
        UserInfoB['TypeExperiment'] = 3

        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20, 30 , 40]:
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
            UserInfoB['simulation'].slicingDim = [2,1]
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)

    def EXP_31_CSFn2_CSFn1_Cascade_finetune(UserInfoB):
        
        # UserInfoB['simulation'].GPU_Index = "0"
        UserInfoB['Model_Method'] = 'Cascade' # , 'HCascade']:
        UserInfoB['upsample'].Scale = 1
        UserInfoB['TypeExperiment'] = 13
        UserInfoB['simulation'].batch_size = 50
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['simulation'].LR_Scheduler = False
        UserInfoB['copy_Thalamus'] = True 
        UserInfoB['lossFunction_Index'] = 4


        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [10, 20 ,30 ,40]:     
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
            # UserInfoB['simulation'].slicingDim = [2,1,0]
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)


    def EXP23a_TL_CSFn2_ResNet2_DiceLoss(UserInfoB):
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['simulation'].FCN1_NLayers = 0
        UserInfoB['copy_Thalamus'] = False
        
        UserInfoB['simulation'].batch_size = 50

        for UserInfoB['simulation'].FCN2_NLayers in [0, 1, 2]:
            
            if UserInfoB['simulation'].FCN2_NLayers == 0: 
                UserInfoB['simulation'].FCN_FeatureMaps = 0          
                for x in [2,1,0]:
                    UserInfoB['simulation'].slicingDim = [x]
                    UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]  
                    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
                    Run(UserInfoB, IV)
                    
            else:
                for UserInfoB['simulation'].FCN_FeatureMaps in [10, 20 , 30 , 40]: 

                    for x in [2,1,0]:
                        UserInfoB['simulation'].slicingDim = [x]
                        UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]  
                        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
                        Run(UserInfoB, IV)

    def EXP23b_TL_CSFn2_ResNet2_DiceLoss(UserInfoB):
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['simulation'].FCN1_NLayers = 1
        UserInfoB['copy_Thalamus'] = False
        
        UserInfoB['simulation'].batch_size = 50

        for UserInfoB['simulation'].FCN2_NLayers in [2]: # 0, 1, 
            for UserInfoB['simulation'].FCN_FeatureMaps in [10, 20 , 30 , 40]:                
                for x in [2,1,0]:
                    UserInfoB['simulation'].slicingDim = [x]
                    UserInfoB['simulation'].nucleus_Index = [1, 2,4,5,6,7,8,9,10,11,12,13,14]  
                    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
                    Run(UserInfoB, IV)

    def EXP23c_TL_CSFn2_ResNet2_DiceLoss(UserInfoB):
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['simulation'].FCN1_NLayers = 2
        UserInfoB['copy_Thalamus'] = False
        
        UserInfoB['simulation'].batch_size = 50

        for UserInfoB['simulation'].FCN2_NLayers in [2]:  # 0, 1, 
            for UserInfoB['simulation'].FCN_FeatureMaps in [10, 20 , 30 , 40]:                
                for x in [2,1,0]:
                    UserInfoB['simulation'].slicingDim = [x]
                    UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]  
                    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
                    Run(UserInfoB, IV)


    def EXP_24_CSFn2_Cascade_finetune(UserInfoB):
        
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['upsample'].Scale = 1
        UserInfoB['TypeExperiment'] = 8
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
            
        UserInfoB['simulation'].LR_Scheduler = False  
        UserInfoB['simulation'].batch_size = 50
        UserInfoB['simulation'].num_Layers = 3
        
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 ,30 ,40]: 
            for x in [2,1,0]:
                UserInfoB['simulation'].slicingDim = [x]
                UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
                IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
                Run(UserInfoB, IV)

    def EXP33_Resnet2_BCEDice_fineTune_ET_Ps_Main_All_folds(UserInfoB):
        
        def predict_Thalamus_For_SD0(UserI):

            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)
        
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 3
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
        UserInfoB['TypeExperiment'] = 15
        UserInfoB['simulation'].LR_Scheduler = True    
        

        for x in ['a','b','c']:
            UserInfoB['CrossVal'].index   = [x]

            UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
            UserInfoB['simulation'].slicingDim = [0]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
            predict_Thalamus_For_SD0(UserInfoB)

            UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30
            UserInfoB['simulation'].slicingDim = [1]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)

            UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
            UserInfoB['simulation'].slicingDim = [2]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)  

    def EXP34_Resnet2_LogEDice_fineTune_ET_Ps_Main_NonCascade(UserInfoB):
        
        def predict_Thalamus_For_SD0(UserI):

            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)
        
        UserInfoB['Model_Method'] = 'normal'
        UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
        UserInfoB['TypeExperiment'] = 15
        UserInfoB['simulation'].LR_Scheduler = True    
        
        UserInfoB['simulation'].ReadAugments_Mode = False 
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20

        for x in ['a','b','c']:
            UserInfoB['CrossVal'].index   = [x]     

            for y in [2,1,0]:
                UserInfoB['simulation'].slicingDim = [y]
                UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
                IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
                Run(UserInfoB, IV)



    def EXP35_CSFn2_Cascade_finetune_All_folds(UserInfoB):
        
        def predict_Thalamus_For_SD0(UserI):

            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)
        
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['upsample'].Scale = 1
        UserInfoB['TypeExperiment'] = 8
        UserInfoB['simulation'].num_Layers = 3
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
            
        UserInfoB['simulation'].LR_Scheduler = False  
        UserInfoB['simulation'].batch_size = 50
        UserInfoB['simulation'].num_Layers = 3
        

        for x in ['b','c','d']:
            UserInfoB['CrossVal'].index   = [x]

            UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30
            UserInfoB['simulation'].slicingDim = [0]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
            predict_Thalamus_For_SD0(UserInfoB)

            UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
            UserInfoB['simulation'].slicingDim = [1]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)

            UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
            UserInfoB['simulation'].slicingDim = [2]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)  

    def EXP36_CSFn2_Cascade_TL_Res_FCN_Unet_full_finetune_All_folds(UserInfoB):
        
        def predict_Thalamus_For_SD0(UserI):

            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)
        
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
        
        UserInfoB['simulation'].batch_size = 50
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20    
        UserInfoB['tag_temp'] = '_Full_FineTune'
        UserInfoB['best_network_MPlanar'] = False

        for x in ['a', 'b','c','d']:
            UserInfoB['CrossVal'].index   = [x]

            UserInfoB['simulation'].slicingDim = [0]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]   
            UserInfoB['simulation'].FCN1_NLayers = 2
            UserInfoB['simulation'].FCN2_NLayers = 1   
            UserInfoB['simulation'].FCN_FeatureMaps = 30 
            predict_Thalamus_For_SD0(UserInfoB)

            UserInfoB['simulation'].slicingDim = [1]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
            UserInfoB['simulation'].FCN1_NLayers = 0
            UserInfoB['simulation'].FCN2_NLayers = 1   
            UserInfoB['simulation'].FCN_FeatureMaps = 10
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)

            UserInfoB['simulation'].slicingDim = [2]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14] 
            UserInfoB['simulation'].FCN1_NLayers = 2
            UserInfoB['simulation'].FCN2_NLayers = 1   
            UserInfoB['simulation'].FCN_FeatureMaps = 40
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)  


    def __test_for_Manoj():
        # UserInfoB['SubExperiment'].Mode_JustThis = True
        # UserInfoB['SubExperiment'].Tag = 'sE13_Cascade_ResFCN_Unet2'

        UserInfoB['Experiments'].Index = '6'
        # UserInfoB['Experiments'].Tag = 'test_Manoj'
        UserInfoB['simulation'].TestOnly = True


        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['copy_Thalamus'] = False
        
        UserInfoB['simulation'].batch_size = 50
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20    
        UserInfoB['CrossVal'].index   = 'b'
        for x in [2,1,0]:
            UserInfoB['simulation'].slicingDim = [x]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
            UserInfoB['simulation'].FCN1_NLayers = 0
            UserInfoB['simulation'].FCN2_NLayers = 0  
            UserInfoB['simulation'].FCN_FeatureMaps = 0
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)

    def EXP37_CSFn2_Cascade_TL_Res_Unet_finetune_All_folds(UserInfoB):
            
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL' 
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
        UserInfoB['simulation'].batch_size = 50
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20    
        UserInfoB['simulation'].FCN1_NLayers = 0
        UserInfoB['simulation'].FCN2_NLayers = 0  
        UserInfoB['simulation'].FCN_FeatureMaps = 0
        UserInfoB['simulation'].LR_Scheduler = False


        UserInfoB['Experiments'].Tag = 'BC_CSFn'

        # applyPreprocess.main(paramFunc.Run(UserInfoB, terminal=True), 'experiment')

        # for x in [2,1,0]:
        #     UserInfoB['simulation'].slicingDim = [x]
        #     UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
        #     IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        #     Run(UserInfoB, IV)

        smallFuncs.apply_MajorityVoting(paramFunc.Run(UserInfoB, terminal=False))

    # def EXP38_WMn_Cascade_Res_FCN_Unet_full_finetune_All_folds(UserInfoB):

    def EXP32_Resnet2_LogDice_fineTune_ET_Ps_Main_OtherFolds(UserInfoB):
        
        def predict_Thalamus_For_SD0(UserI):

            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)

            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
            IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
            Run(UserI, IV)
        
        UserInfoB['Model_Method'] = 'Cascade'
        UserInfoB['simulation'].num_Layers = 3
        # UserInfoB['simulation'].slicingDim = [2,1,0]
        UserInfoB['architectureType'] = 'Res_Unet2'
        UserInfoB['lossFunction_Index'] = 4
        # UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
        UserInfoB['TypeExperiment'] = 15
        UserInfoB['simulation'].LR_Scheduler = True    
        

        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
        UserInfoB['simulation'].slicingDim = [0]
        UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
        # IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        predict_Thalamus_For_SD0(UserInfoB)

        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30
        UserInfoB['simulation'].slicingDim = [1]
        UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        Run(UserInfoB, IV)

        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        UserInfoB['simulation'].slicingDim = [2]
        UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        Run(UserInfoB, IV)    

    def EXP38_CSFn2_Cascade_TL_Res_Unet_finetune_other_permutations(UserInfoB):
            
        UserInfoB['CrossVal'].index   = ['a']

        UserInfoB['permutation_Index'] = 0
        UserInfoB['TypeExperiment'] = 11
        UserInfoB['Model_Method'] = 'Cascade' 
        UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
        UserInfoB['lossFunction_Index'] = 4
        UserInfoB['Experiments'].Index = '6'
        UserInfoB['copy_Thalamus'] = False
        UserInfoB['simulation'].batch_size = 50
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20    

        for x in [2]: # ,1,0]:
            UserInfoB['simulation'].slicingDim = [x]
            UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
            UserInfoB['simulation'].FCN1_NLayers = 0
            UserInfoB['simulation'].FCN2_NLayers = 0  
            UserInfoB['simulation'].FCN_FeatureMaps = 0
            IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
            Run(UserInfoB, IV)

def EXP_CSFn_test_new_Cases(UserInfoB):
        
    UserInfoB['TypeExperiment'] = 11
    UserInfoB['Model_Method'] = 'Cascade' 
    UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL' # ''
    UserInfoB['lossFunction_Index'] = 4
    UserInfoB['copy_Thalamus'] = False
    UserInfoB['simulation'].batch_size = 50
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20    
    UserInfoB['simulation'].FCN1_NLayers = 0
    UserInfoB['simulation'].FCN2_NLayers = 0  
    UserInfoB['simulation'].FCN_FeatureMaps = 0

    applyPreprocess.main(paramFunc.Run(UserInfoB, terminal=True), 'experiment')

    for x in [2,1,0]:
        UserInfoB['simulation'].slicingDim = [x]
        UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
        IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
        Run(UserInfoB, IV)

    smallFuncs.apply_MajorityVoting(paramFunc.Run(UserInfoB, terminal=False))

def EXP_WMn_test_new_Cases(UserInfoB):
    
    def predict_Thalamus_For_SD0(UserI):

        UserI['simulation'].slicingDim = [2]
        UserI['simulation'].nucleus_Index = [1]
        IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
        Run(UserI, IV)

        UserI['simulation'].slicingDim = [0]
        UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
        IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
        Run(UserI, IV)
    
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
    UserInfoB['copy_Thalamus'] = False
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
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
    Run(UserInfoB, IV)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].slicingDim = [2]
    UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
    Run(UserInfoB, IV)    

    
    merge_results_and_apply_25D(UserInfoB)

def Run_Csfn_with_Best_WMn_architecture(UserInfoB):
    
    def predict_Thalamus_For_SD0(UserI):

        UserI['simulation'].slicingDim = [2]
        UserI['simulation'].nucleus_Index = [1]
        IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
        Run(UserI, IV)

        UserI['simulation'].slicingDim = [0]
        UserI['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
        IV = InitValues( UserI['simulation'].nucleus_Index , UserI['simulation'].slicingDim)
        Run(UserI, IV)
    
    def merge_results_and_apply_25D(UserInfoB):

        UserInfoB['best_network_MPlanar'] = True
        params = paramFunc.Run(UserInfoB, terminal=True)
        Directory = params.WhichExperiment.Experiment.address + '/results'
        Output = "sE12_Cascade_FM00_Res_Unet2_NL3_LS_MyLogDice_US1_CSFn2_Init_Main_wBiasCorrection_CV_%s"%(UserInfoB['CrossVal'].index[0])
        os.system("mkdir %s; cd %s; mkdir sd0 sd1 sd2"%(Directory + '/' + Output, Directory + '/' + Output))
        os.system("cp -r %s/sE12_Cascade_FM40_Res_Unet2_NL3_LS_MyLogDice_US1_CSFn2_Init_Main_wBiasCorrection_CV_%s/sd0/vimp* %s/sd0/"%(Directory, UserInfoB['CrossVal'].index[0] , Directory +'/'+ Output))
        os.system("cp -r %s/sE12_Cascade_FM30_Res_Unet2_NL3_LS_MyLogDice_US1_CSFn2_Init_Main_wBiasCorrection_CV_%s/sd1/vimp* %s/sd1/"%(Directory, UserInfoB['CrossVal'].index[0] , Directory +'/'+ Output))
        os.system("cp -r %s/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyLogDice_US1_CSFn2_Init_Main_wBiasCorrection_CV_%s/sd2/vimp* %s/sd2/"%(Directory, UserInfoB['CrossVal'].index[0] , Directory +'/'+ Output))

        smallFuncs.apply_MajorityVoting(params)

    UserInfoB['Model_Method'] = 'Cascade'
    UserInfoB['simulation'].num_Layers = 3
    UserInfoB['architectureType'] = 'Res_Unet2'
    UserInfoB['lossFunction_Index'] = 4
    UserInfoB['Experiments'].Index = '7'
    UserInfoB['copy_Thalamus'] = False
    UserInfoB['TypeExperiment'] = 8
    UserInfoB['simulation'].LR_Scheduler = False    
    

    applyPreprocess.main(paramFunc.Run(UserInfoB, terminal=True), 'experiment')
    
    
    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
    UserInfoB['simulation'].slicingDim = [0]
    UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
    predict_Thalamus_For_SD0(UserInfoB)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30
    UserInfoB['simulation'].slicingDim = [1]
    UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
    Run(UserInfoB, IV)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].slicingDim = [2]
    UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]       
    IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
    Run(UserInfoB, IV)    

    
    merge_results_and_apply_25D(UserInfoB)

   
# UserInfoB['simulation'].ReadAugments_Mode = False

Run_Csfn_with_Best_WMn_architecture(UserInfoB)

# UserInfoB['simulation'].ReadAugments_Mode = False 
# smallFuncs.apply_MajorityVoting(paramFunc.Run(UserInfoB, terminal=False)) 
 # smallFuncs.Extra_mergingResults()


