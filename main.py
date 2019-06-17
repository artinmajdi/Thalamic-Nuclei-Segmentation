import os
import sys
sys.path.append(os.path.dirname(__file__))  # sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
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
                
def Run(UserInfoB,InitValues):

    def print_FullExp(IV):
        print('---------------------------------------------------------------')
        print('Full Experiment Info')
        print('slicingDim' , IV.slicingDim , 'Nuclei_Indexes' , IV.Nuclei_Indexes , 'GPU:  ', UserInfoB['simulation'].GPU_Index, UserInfoB['Model_Method'] , '\n')
        print('---------------------------------------------------------------')
    print_FullExp(InitValues)

    def HierarchicalStages_single_Class(UserInfoB):

        BB = smallFuncs.Nuclei_Class(1,'HCascade')

        print('************ stage 1 ************')
        if 1 in InitValues.Nuclei_Indexes: 
            UserInfoB['simulation'].nucleus_Index = [1]
            Run_Main(UserInfoB)

        print('************ stage 2 ************')                    
        for UserInfoB['simulation'].nucleus_Index in BB.HCascade_Parents_Identifier(InitValues.Nuclei_Indexes):
            Run_Main(UserInfoB)

        print('************ stage 3 ************')
        for UserInfoB['simulation'].nucleus_Index in BB.remove_Thalamus_From_List(InitValues.Nuclei_Indexes):
            Run_Main(UserInfoB)

    def HierarchicalStages_Multi_Class(UserInfoB):

        BB = smallFuncs.Nuclei_Class(1,'HCascade')

        print('************ stage 1 ************')
        if 1 in InitValues.Nuclei_Indexes: 
            UserInfoB['simulation'].nucleus_Index = [1]
            Run_Main(UserInfoB)

        print('************ stage 2 ************')                    
        UserInfoB['simulation'].nucleus_Index = [1.1, 1.2, 1.3, 2]
        Run_Main(UserInfoB)

        print('************ stage 3 ************')
        for parent in [1.1, 1.2, 1.3]:
            CC = smallFuncs.Nuclei_Class(parent,'HCascade')
            UserInfoB['simulation'].nucleus_Index = CC.child
            Run_Main(UserInfoB)

    def Cascade_Loop(UserInfoB):

        if not UserInfoB['simulation'].Multi_Class_Mode:
            for UserInfoB['simulation'].nucleus_Index in InitValues.Nuclei_Indexes:
                Run_Main(UserInfoB)

        else:
            # if (1 in UserInfoB['simulation'].nucleus_Index) and (UserInfoB['Model_Method'] != 'FCN'):
            UserInfoB['simulation'].nucleus_Index = [1]
            Run_Main(UserInfoB)

            BB = smallFuncs.Nuclei_Class(1,'Cascade')            
            UserInfoB['simulation'].nucleus_Index = BB.remove_Thalamus_From_List(list(BB.All_Nuclei().Indexes))
            Run_Main(UserInfoB)

    def Run_Main(UserInfoB):
        def subRun(UserInfoB):             
            params = paramFunc.Run(UserInfoB, terminal=False)

            print('---------------------------------------------------------------')
            print(' Nucleus:', UserInfoB['simulation'].nucleus_Index  , ' | GPU:', UserInfoB['simulation'].GPU_Index , ' | SD',sd, \
                ' | Dropout', UserInfoB['DropoutValue'] , ' | LR' , UserInfoB['simulation'].Learning_Rate, ' | NL' , UserInfoB['simulation'].num_Layers,\
                ' | ', UserInfoB['Model_Method'] , '|  FM', UserInfoB['simulation'].FirstLayer_FeatureMap_Num)

            print('Experiment:', params.WhichExperiment.Experiment.name)                              
            print('SubExperiment:', params.WhichExperiment.SubExperiment.name)
            print('---------------------------------------------------------------')

            if 1 in UserInfoB['simulation'].nucleus_Index:
                input_model  = params.WhichExperiment.Experiment.address + '/models/' + params.WhichExperiment.SubExperiment.name_Thalmus_network
                output_model = params.WhichExperiment.Experiment.address + '/models/' + params.WhichExperiment.SubExperiment.name
                os.system('mkdir %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

                input_model  = params.WhichExperiment.Experiment.address + '/results/' + params.WhichExperiment.SubExperiment.name_Thalmus_network
                output_model = params.WhichExperiment.Experiment.address + '/results/' + params.WhichExperiment.SubExperiment.name
                os.system('mkdir %s ; cp -r %s/* %s/'%(output_model , input_model , output_model))

            """
            elif (1.4 in UserInfoB['simulation'].nucleus_Index) and (not UserInfoB['simulation'].Multi_Class_Mode):
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
                save_Anteior_BBox(params)

            else:
                Data, params = datasets.loadDataset(params) 
                print(Data.Train.Image.shape)               
                choosingModel.check_Run(params, Data)              
                K.clear_session()
            """
                    
        for sd in InitValues.slicingDim:            
            if not (sd == 0 and UserInfoB['simulation'].nucleus_Index == 1):
                UserInfoB['simulation'].slicingDim = [sd]
                subRun(UserInfoB)
                 
    if   UserInfoB['Model_Method'] == 'HCascade':  HierarchicalStages_Multi_Class(UserInfoB)
    elif UserInfoB['Model_Method'] == 'Cascade' :  Cascade_Loop(UserInfoB)
    elif UserInfoB['Model_Method'] == 'singleRun': Run_Main(UserInfoB)
    else: Cascade_Loop(UserInfoB)
     
def preMode(UserInfoB):
    UserInfoB = smallFuncs.terminalEntries(UserInfoB)
    params = paramFunc.Run(UserInfoB, terminal=False)   
    datasets.movingFromDatasetToExperiments(params)
    applyPreprocess.main(params, 'experiment')
    K = smallFuncs.gpuSetting(params.WhichExperiment.HardParams.Machine.GPU_Index)
    return UserInfoB, K

UserInfoB, K = preMode(UserInfo.__dict__)

IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)


for UserInfoB['upsample'].Scale in [2 , 4]:
    for UserInfoB['simulation'].num_Layers in [2 , 3 , 4]:
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30]:
            Run(UserInfoB, IV)

            
for UserInfoB['upsample'].Scale in [2 , 4]:
    for UserInfoB['simulation'].num_Layers in [5]:
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30]:
            Run(UserInfoB, IV)

for UserInfoB['upsample'].Scale in [4]:
    for UserInfoB['simulation'].num_Layers in [6]:
        for UserInfoB['simulation'].FirstLayer_FeatureMap_Num in [20 , 30]:
            Run(UserInfoB, IV)

K.clear_session()
