import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import modelFuncs.LossFunction as LossFunction
import modelFuncs.Metrics as Metrics
import modelFuncs.Optimizers as Optimizers
# from Parameters import Classes
import otherFuncs.smallFuncs as smallFuncs
from otherFuncs import datasets
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
import json

"""
def temp_Experiments_preSet(UserInfoB):

    # TypeExperiment == 1: # Main
    # TypeExperiment == 2: # Transfer Learn ET
    # TypeExperiment == 3: # SRI
    # TypeExperiment == 4: # Predict ET from MS&Ctrl
    # TypeExperiment == 5: # Train ET Initialized from 3T
    # TypeExperiment == 6: # Train Main+ET
    # TypeExperiment == 7: # Train Main+ET+SRI
    # TypeExperiment == 8: # Train Main+SRI
    # TypeExperiment == 9: # Train ET Initialized from Main+SRI
    # TypeExperiment == 10: # Main + All Augments
    # TypeExperiment == 11: # Main + Init from Thalamus
    # TypeExperiment == 12: # Main + Init from 3T
    class TypeExperimentFuncs():
        def __init__(self):            
            class ReadTrainC:
                def __init__(self, SRI=0 , ET=0 , Main=1 , CSFn=0):   
                    class readAugments: Mode, Tag, LoadAll = True, '', False
                    self.SRI  = SRI  > 0.5
                    self.ET   = ET   > 0.5
                    self.Main = Main > 0.5
                    self.CSFn = CSFn > 0.5

                    self.ReadAugments = readAugments                    
            self.ReadTrainC = ReadTrainC

            class Transfer_LearningC:
                def __init__(self, Mode=False , FrozenLayers = [0] , Tag = '_TF' , Stage = 0):
                    self.Mode         = Mode
                    self.FrozenLayers = FrozenLayers
                    self.Stage        = Stage
                    self.Tag          = Tag
            self.Transfer_LearningC = Transfer_LearningC

        def main(self, TypeExperiment = 1):
            switcher = {
                1:  (11  ,   self.ReadTrainC(SRI=0 , ET=0 , Main=1)  ,  self.Transfer_LearningC() ),
                2:  (11  ,   self.ReadTrainC(SRI=0 , ET=1 , Main=0)  ,  self.Transfer_LearningC(Mode=True  , FrozenLayers=[0] , Tag = '_TF') ),
                3:  (8   ,   self.ReadTrainC(SRI=1 , ET=0 , Main=0)  ,  self.Transfer_LearningC() ),
                4:  (11  ,   self.ReadTrainC(SRI=0 , ET=1 , Main=0)  ,  self.Transfer_LearningC() ),
                5:  (11  ,   self.ReadTrainC(SRI=0 , ET=1 , Main=0)  ,  self.Transfer_LearningC() ),
                6:  (11  ,   self.ReadTrainC(SRI=0 , ET=1 , Main=1)  ,  self.Transfer_LearningC() ),
                7:  (11  ,   self.ReadTrainC(SRI=1 , ET=1 , Main=1)  ,  self.Transfer_LearningC() ),
                8:  (11  ,   self.ReadTrainC(SRI=1 , ET=0 , Main=1)  ,  self.Transfer_LearningC() ),
                9:  (11  ,   self.ReadTrainC(SRI=0 , ET=1 , Main=0)  ,  self.Transfer_LearningC() ),
                10: (11  ,   self.ReadTrainC(SRI=0 , ET=0 , Main=1)  ,  self.Transfer_LearningC() ),
                11: (11  ,   self.ReadTrainC(SRI=0 , ET=0 , Main=1)  ,  self.Transfer_LearningC() ),
                12: (11  ,   self.ReadTrainC(SRI=0 , ET=0 , Main=1)  ,  self.Transfer_LearningC() ),
                }
            return switcher.get(TypeExperiment , 'wrong Index')

    def extra_info(UserInfoB):
        if UserInfoB['TypeExperiment'] == 4:
            UserInfoB['simulation'].TestOnly = True

        elif UserInfoB['TypeExperiment'] == 3:
            UserInfoB['InitializeB'].FromThalamus   = True
            UserInfoB['InitializeB'].FromOlderModel = True
            UserInfoB['InitializeB'].From_3T        = False  

        elif UserInfoB['TypeExperiment'] == 5: 
            UserInfoB['InitializeB'].FromThalamus   = False
            UserInfoB['InitializeB'].FromOlderModel = False
            UserInfoB['InitializeB'].From_3T        = True  
            UserInfoB['simulation'].TestOnly        = False  
        
        elif UserInfoB['TypeExperiment'] == 7:
            UserInfoB['SubExperiment'].Tag += '_Main_PlusET_PlusSRI'

        #elif UserInfoB['TypeExperiment'] == 8:
        #    UserInfoB['SubExperiment'].Tag = '_Main_PlusSRI'

        elif UserInfoB['TypeExperiment'] == 9:
            UserInfoB['InitializeB'].FromThalamus   = False
            UserInfoB['InitializeB'].FromOlderModel = True
            UserInfoB['InitializeB'].From_3T        = False   
            # UserInfoB['CrossVal'].Mode = False
            
        # elif UserInfoB['TypeExperiment'] == 10:
        #     UserInfoB['ReadTrain'].ReadAugments.LoadAll = True 
            # UserInfoB['SubExperiment'].Tag += '_Main_AllAugments'   
            
        elif UserInfoB['TypeExperiment'] == 11:
            UserInfoB['SubExperiment'].Tag += '_Main_Init_FromThalamus'  
            UserInfoB['InitializeB'].FromThalamus   = True
            UserInfoB['InitializeB'].FromOlderModel = False
            UserInfoB['InitializeB'].From_3T        = False       

        elif UserInfoB['TypeExperiment'] == 12:
            UserInfoB['SubExperiment'].Tag += '_Main_Init_From3T'  
            UserInfoB['InitializeB'].FromThalamus   = False
            UserInfoB['InitializeB'].FromOlderModel = False
            UserInfoB['InitializeB'].From_3T        = True  

        return UserInfoB

    a,b,c = TypeExperimentFuncs().main(UserInfoB['TypeExperiment'])
    UserInfoB['SubExperiment'].Index = a
    UserInfoB['ReadTrain']           = b
    UserInfoB['Transfer_Learning']   = c
    UserInfoB = extra_info(UserInfoB)

    return UserInfoB
"""
def temp_Experiments_preSet_V2(UserInfoB):

    class TypeExperimentFuncs():
        def __init__(self):            
            class ReadTrainC:
                def __init__(self, SRI=0 , ET=0 , Main=0 , CSFn=0):   
                    # class readAugments: Mode, Tag, LoadAll = False, '', False  # temp
                    class readAugments: Mode, Tag, LoadAll = True, '', False
                    self.SRI  = SRI  > 0.5
                    self.ET   = ET   > 0.5
                    self.Main = Main > 0.5
                    self.CSFn = CSFn > 0.5

                    self.ReadAugments = readAugments                    
            self.ReadTrainC = ReadTrainC

            class Transfer_LearningC:
                def __init__(self, Mode=False , FrozenLayers = [0] , Tag = '_TF' , Stage = 0):
                    self.Mode         = Mode
                    self.FrozenLayers = FrozenLayers
                    self.Stage        = Stage
                    self.Tag          = Tag
            self.Transfer_LearningC = Transfer_LearningC

            class InitializeB:
                def __init__(self, FromThalamus=False , FromOlderModel=False , From_3T=False , From_7T=False , From_CSFn=False):
                    self.FromThalamus   = FromThalamus
                    self.FromOlderModel = FromOlderModel
                    self.From_3T        = From_3T
                    self.From_7T        = From_7T
                    self.From_CSFn      = From_CSFn
            self.InitializeB = InitializeB

        def main(self, TypeExperiment = 1):
            switcher = {
                1:  (8   ,   self.ReadTrainC(SRI=1)          , self.InitializeB()                    ,  self.Transfer_LearningC() ),
                2:  (11  ,   self.ReadTrainC(Main=1)         , self.InitializeB(From_3T=True)        ,  self.Transfer_LearningC() ),
                3:  (11  ,   self.ReadTrainC(ET=1)           , self.InitializeB(From_7T=True)        ,  self.Transfer_LearningC() ),
                4:  (11  ,   self.ReadTrainC(ET=1)           , self.InitializeB()                    ,  self.Transfer_LearningC(Mode=True  , FrozenLayers=[0] , Tag = '_TF') ),
                5:  (11  ,   self.ReadTrainC(ET=1)           , self.InitializeB()                    ,  self.Transfer_LearningC() ),
                6:  (11  ,   self.ReadTrainC(SRI=1, Main=1)  , self.InitializeB()                    ,  self.Transfer_LearningC() ),
                7:  (11  ,   self.ReadTrainC(ET=1)           , self.InitializeB()                    ,  self.Transfer_LearningC() ),
                8:  (11  ,   self.ReadTrainC(Main=1 , SRI=1) , self.InitializeB(FromThalamus=True)   ,  self.Transfer_LearningC() ),
                9:  (11  ,   self.ReadTrainC(Main=1 , SRI=1) , self.InitializeB(From_3T=True)        ,  self.Transfer_LearningC() ),
                10:  (12  ,   self.ReadTrainC(CSFn=1)        , self.InitializeB(From_7T=True)        ,  self.Transfer_LearningC() ),
                11:  (12  ,   self.ReadTrainC(Main=1 , SRI=1) , self.InitializeB(From_3T=True)       ,  self.Transfer_LearningC() ),
                12:  (12  ,   self.ReadTrainC(CSFn=1)         , self.InitializeB(From_3T=True)       ,  self.Transfer_LearningC() ),
                13:  (12  ,   self.ReadTrainC(ET=1)           , self.InitializeB(From_7T=True)       ,  self.Transfer_LearningC() ),
                14:  (12  ,   self.ReadTrainC(CSFn=1)         , self.InitializeB(From_CSFn=True)     ,  self.Transfer_LearningC() ),
                }
            return switcher.get(TypeExperiment , 'wrong Index')

    a,b,c,d = TypeExperimentFuncs().main(UserInfoB['TypeExperiment'])
    UserInfoB['SubExperiment'].Index = a
    UserInfoB['ReadTrain']           = b
    UserInfoB['Transfer_Learning']   = d
    UserInfoB['InitializeB']         = c
    if UserInfoB['TypeExperiment'] == 5: UserInfoB['simulation'].TestOnly = True
    if UserInfoB['TypeExperiment'] == 2: UserInfoB['SubExperiment'].Tag = '_Main_Init_3T_AllAugs' # _250epochs_Wo_LR_scheduler
    if UserInfoB['TypeExperiment'] == 3: UserInfoB['SubExperiment'].Tag = '_ET_Init_Main_AllAugs'
    if UserInfoB['TypeExperiment'] == 7: UserInfoB['SubExperiment'].Tag = '_ET_Init_Rn_AllAugs'
    if UserInfoB['TypeExperiment'] == 8: UserInfoB['SubExperiment'].Tag = '_Main_PlusSRI_InitFrom_Th'
    if UserInfoB['TypeExperiment'] == 9: UserInfoB['SubExperiment'].Tag = '_Main_PlusSRI_InitFrom_3T' 
    if UserInfoB['TypeExperiment'] == 10: UserInfoB['SubExperiment'].Tag = '_CSFn__Init_Main'
    if UserInfoB['TypeExperiment'] == 11: UserInfoB['SubExperiment'].Tag = '_Main_Plus_3T_InitFrom_3T_NoSchedular'
    if UserInfoB['TypeExperiment'] == 12: UserInfoB['SubExperiment'].Tag = '_CSFn__Init_3T' # _reversed_Contrast
    if UserInfoB['TypeExperiment'] == 13: UserInfoB['SubExperiment'].Tag = '_ET_InitFrom_3Tp7T_NoSchedular' # _WeightedClass'
    if UserInfoB['TypeExperiment'] == 14: UserInfoB['SubExperiment'].Tag = '_CSFn__Init_THOMAS_CSFn' 

    # if UserInfoB['TypeExperiment'] == 5: UserInfoB['simulation'].TestOnly = True
    # if UserInfoB['TypeExperiment'] == 2: UserInfoB['SubExperiment'].Tag = '_Main_Init_3T' # _250epochs_Wo_LR_scheduler
    # if UserInfoB['TypeExperiment'] == 3: UserInfoB['SubExperiment'].Tag = '_ET_Init_Main'
    # if UserInfoB['TypeExperiment'] == 7: UserInfoB['SubExperiment'].Tag = '_ET_Init_Rn'
    # if UserInfoB['TypeExperiment'] == 8: UserInfoB['SubExperiment'].Tag = '_Main_PlusSRI_InitFrom_Th'
    # if UserInfoB['TypeExperiment'] == 9: UserInfoB['SubExperiment'].Tag = '_Main_PlusSRI_InitFrom_3T' 
    # if UserInfoB['TypeExperiment'] == 10: UserInfoB['SubExperiment'].Tag = '_CSFn__Init_Main'
    # if UserInfoB['TypeExperiment'] == 11: UserInfoB['SubExperiment'].Tag = '_Main_Plus_3T_InitFrom_3T'
    # if UserInfoB['TypeExperiment'] == 12: UserInfoB['SubExperiment'].Tag = '_CSFn__Init_3T' # _reversed_Contrast
    # if UserInfoB['TypeExperiment'] == 13: UserInfoB['SubExperiment'].Tag = '_ET_InitFrom_3Tp7T' # _WeightedClass'
    # if UserInfoB['TypeExperiment'] == 14: UserInfoB['SubExperiment'].Tag = '_CSFn__Init_THOMAS_CSFn' 

    return UserInfoB

def Run(UserInfoB, terminal=False):
        
    if terminal: UserInfoB = smallFuncs.terminalEntries(UserInfoB)

    UserInfoB = temp_Experiments_preSet_V2(UserInfoB)

    class params:
        WhichExperiment = func_WhichExperiment(UserInfoB)
        preprocess      = func_preprocess(UserInfoB)
        Augment         = func_Augment(UserInfoB) 
        directories     = smallFuncs.search_ExperimentDirectory(WhichExperiment)
        UserInfo        = UserInfoB

    return params

def func_Exp_subExp_Names(UserInfo):

    def func_subExperiment():
        
        FM = UserInfo['simulation'].FirstLayer_FeatureMap_Num
        DO = UserInfo['DropoutValue']
        SE = UserInfo['SubExperiment']
        NL = UserInfo['simulation'].num_Layers
        AT =  '_' + UserInfo['architectureType']
        method = UserInfo['Model_Method']        
        # def field_Strength_Tag():
        #     if UserInfo['ReadTrain'].SRI:                                 return '_3T'    
        #     elif UserInfo['ReadTrain'].Main or UserInfo['ReadTrain'].ET:  return '_7T' 
        #     else:                                                         return '_CSFn'                                                                        
               
        class subExperiment:
            def __init__(self, tag):                
                self.index = SE.Index
                self.tag = tag
                self.name_thalamus = ''            
                self.name = 'sE' + str(SE.Index) +  '_' + self.tag            
                self.name_Init_from_3T = 'sE8_' + method + '_FM' + str(FM) + AT 
                #self.name_Init_from_7T = 'sE11_' + method + '_FM' + str(FM)
                self.name_Init_from_7T = 'sE12_' + method + '_FM' + str(FM) + AT # + '_3T7T'
                self.name_Init_from_CSFn = 'sE9_' + method + '_FM' + str(FM) + AT  
                self.name_Thalmus_network = 'sE8_Predictions_Full_THALAMUS' # sE8_FM20_U-Net4_1-THALMAUS 
                self.crossVal = UserInfo['CrossVal']()

        # tag = method + '_FM' + str(FM) + '_DO' + str(DO) + AT + SE.Tag   
        
        tag = method + '_FM' + str(FM) + AT + SE.Tag
        # if  'FCN' in UserInfo['architectureType']:
        tag += '_NL' + str(NL) # '_' + UserInfo['normalize'].Method  

        if UserInfo['lossFunction_Index'] != 1: 
            _, a = LossFunction.LossInfo(UserInfo['lossFunction_Index'])
            tag += '_' + a

        if not UserInfo['simulation'].Weighted_Class_Mode: tag += '_NotWeighted'
        else: tag += '_Weighted'

        if UserInfo['simulation'].Multi_Class_Mode: tag += '_MultiClass'
        else: tag += '_SingleClass'

        if UserInfo['upsample'].Mode: tag += '_Upsampled' + str(UserInfo['upsample'].Scale)
        # tag += '_normalize_On_AllSubjs'

        if UserInfo['CrossVal'].Mode and SE.Index not in [8,9]: tag += '_CV_' + UserInfo['CrossVal'].index[0]
        A = subExperiment(tag)
        print('Init From 3T Tag'  , A.name_Init_from_3T)
        print('Init From 7T Tag'  , A.name_Init_from_7T)
        print('Init From CSFn Tag', A.name_Init_from_CSFn)
        return A

    def func_Experiment():
        EX = UserInfo['Experiments']
        class Experiment:
            index = EX.Index
            tag   = EX.Tag
            name  = 'exp' + str(EX.Index) + '_' + EX.Tag if EX.Tag else 'exp' + str(EX.Index)
            address = smallFuncs.mkDir(UserInfo['Experiments_Address'] + '/' + name)
            PreSet_Experiment_Info_Index = UserInfo['TypeExperiment']
        return Experiment()

    return func_Experiment(), func_subExperiment()  

def func_WhichExperiment(UserInfo):
    
    def WhichExperiment_Class():

        def HardParamsFuncs():
            def ArchtiectureParams():
                class dropout:
                    Mode = True
                    Value = 0.2

                class kernel_size:
                    conv = (3,3)
                    convTranspose = (2,2)
                    output = (1,1)

                class activation:
                    layers = 'relu'
                    output = 'sigmoid'

                class convLayer:
                    # strides = (1,1)
                    Kernel_size = kernel_size()
                    padding = 'SAME' # valid

                class multiclass:
                    num_classes = ''
                    Mode = False

                class maxPooling:
                    strides = (2,2)
                    pool_size = (2,2)

                class method:
                    Type = ''
                    # InitializeFromReference = True # from 3T or WMn for CSFn
                    ReferenceMask = ''
                    havingBackGround_AsExtraDimension = True
                    InputImage2Dvs3D = 2
                    Multiply_By_Thalmaus = False
                    Multiply_By_Rest_For_AV = False
                    save_Best_Epoch_Model = False
                    Use_Coronal_Thalamus_InSagittal = False
                    Use_TestCases_For_Validation = False
                    ImClosePrediction = False

                return dropout, activation, convLayer, multiclass, maxPooling, method

            dropout, activation, convLayer, multiclass, maxPooling, method = ArchtiectureParams()

            class transfer_Learning:
                Mode = False
                Stage = 0 # 1
                FrozenLayers = [0]
                Tag = '_TF'
            
            class classWeight:
                Weight = {0:1 , 1:1}
                Mode = False
            class layer_Params:
                FirstLayer_FeatureMap_Num = 64
                batchNormalization = True
                ConvLayer = convLayer()
                MaxPooling = maxPooling()
                Dropout = dropout()
                Activitation = activation()
                class_weight = classWeight()

            class InitializeB:
                FromThalamus   = False
                FromOlderModel = False
                From_3T        = False  
                From_7T = False  

            class dataGenerator:
                mode = False
                NumSubjects_Per_batch = 5

            class upsample:
                Mode = False
                Scale = 2
            class model:
                architectureType = 'U-Net'
                epochs = ''
                batch_size = ''
                loss = ''
                metrics = ''
                optimizer = ''  # adamax Nadam Adadelta Adagrad  optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                
                verbose = 1
                num_Layers = ''
                InputDimensions = ''
                Layer_Params = layer_Params()
                showHistory = True
                LabelMaxValue = 1                
                Measure_Dice_on_Train_Data = True
                MultiClass = multiclass()
                Initialize = InitializeB()
                Method = method()
                paddingErrorPatience = 20
                Transfer_Learning = transfer_Learning()
                DataGenerator = dataGenerator()
                Upsample = upsample()
                
                

            lossFunction_Index = 1
            model.loss, _ = LossFunction.LossInfo(lossFunction_Index)

            class machine:
                WhichMachine = 'server'
                GPU_Index = ''

            class image:
                # SlicingDirection = 'axial'.lower()
                SaveMode = 'nifti'.lower()

            class template:
                Image = ''
                Mask  = ''

            class hardParams:
                Model    = model
                Template = template()
                Machine  = machine()
                Image    = image()

            return hardParams

        hardParams = HardParamsFuncs()

        class experiment:
            index = ''
            tag = ''
            name = ''
            address = ''
        
        class CrossVal:
            Mode = False
            index = ['a']
            # All_Indexes = ['a' , 'b']
        class subExperiment:
            index = ''
            tag = ''
            name = ''
            name_thalamus = ''
            crossVal = CrossVal()
            name_Init_from_7T = ''
            name_Init_from_3T = ''

        def datasetFunc():
            class validation:
                percentage = 0.1
                fromKeras = False

            class testDs:
                mode = 'percentage' # 'names'
                percentage = 0.3
                subjects = ''

            if 'names' in testDs.mode:
                testDs.subjects = list([''])

            class slicingDirection:
                slicingOrder = [0,1,2]
                slicingOrder_Reverse = [0,1,2]
                slicingDim = 2

            class inputPadding:
                Automatic = True
                HardDimensions = ''

            class hDF5:
                mode = False
                mode_saveTrue_LoadFalse = True

            class readAugmentFn:
                Mode = False
                Tag = ''
                LoadAll = False

            class readTrain:
                Main = True
                ET   = True
                SRI  = True
                CSFn = False
                ReadAugments = readAugmentFn()

            class dataset:
                name = ''
                address = ''
                Validation = validation()
                Test = testDs()
                check_vimp_SubjectName = True
                randomFlag = True
                slicingInfo = slicingDirection()
                gapDilation = 5
                gapOnSlicingDimention = 2
                InputPadding = inputPadding()
                ReadTrain = readTrain()
                HDf5 = hDF5()

            # Dataset_Index = 4
            # dataset.name, dataset.address = datasets.DatasetsInfo(Dataset_Index)
            return dataset

        dataset = datasetFunc()

        class nucleus:
            Organ = 'THALAMUS'
            name = ''
            name_Thalamus = ''
            FullIndexes = ''
            Index = ''

        class WhichExperiment:
            Experiment    = experiment()
            SubExperiment = subExperiment()
            address = ''
            Nucleus = nucleus()
            HardParams = hardParams()
            Dataset = dataset()

        return WhichExperiment()
    WhichExperiment = WhichExperiment_Class()

    def func_Nucleus(MultiClassMode):
        def Experiment_Nucleus_Name_MClass(NucleusIndex, MultiClassMode):
            if len(NucleusIndex) == 1 or not MultiClassMode:
                NucleusName , _, _ = smallFuncs.NucleiSelection( NucleusIndex[0] )
            else:
                if (UserInfo['Model_Method'] == 'HCascade') and (1.1 in UserInfo['simulation'].nucleus_Index):
                    NucleusName = 'MultiClass_HCascade_Groups'
                else:
                    NucleusName = ('MultiClass_' + str(NucleusIndex)).replace(', ','').replace('[','').replace(']','')
                
                    

            return NucleusName

        nucleus_Index = UserInfo['simulation'].nucleus_Index if isinstance(UserInfo['simulation'].nucleus_Index,list) else [UserInfo['simulation'].nucleus_Index]
        class nucleus:
            Organ = 'THALAMUS'
            name = Experiment_Nucleus_Name_MClass(nucleus_Index , MultiClassMode )
            name_Thalamus, FullIndexes, _ = smallFuncs.NucleiSelection( 1 )
            Index = nucleus_Index

        return nucleus

    def func_Dataset():

        def Augment_Tag():
            readAugmentTag = ''
            if UserInfo['Augment_Rotation'].Mode: 
                readAugmentTag = 'wRot'   + str(UserInfo['Augment_Rotation'].AngleMax) + 'd'
            elif UserInfo['Augment_Shear'].Mode:  
                readAugmentTag = 'wShear' + str(UserInfo['Augment_Shear'].ShearMax)
            return readAugmentTag
                
        Dataset = WhichExperiment.Dataset
        def slicingInfoFunc():
            class slicingInfo:
                slicingOrder = ''
                slicingOrder_Reverse = ''
                slicingDim = UserInfo['simulation'].slicingDim[0]

            if slicingInfo.slicingDim == 0:
                slicingInfo.slicingOrder         = [1,2,0]
                slicingInfo.slicingOrder_Reverse = [2,0,1]
            elif slicingInfo.slicingDim == 1:
                slicingInfo.slicingOrder         = [2,0,1]
                slicingInfo.slicingOrder_Reverse = [1,2,0]
            else:
                slicingInfo.slicingOrder         = [0,1,2]
                slicingInfo.slicingOrder_Reverse = [0,1,2]

            return slicingInfo

        Dataset.ReadTrain = UserInfo['ReadTrain']
        Dataset.ReadTrain.ReadAugments.Tag = Augment_Tag()

        Dataset.gapDilation = UserInfo['gapDilation']
        Dataset.HDf5.mode_saveTrue_LoadFalse = UserInfo['mode_saveTrue_LoadFalse']
        Dataset.slicingInfo = slicingInfoFunc()

        Dataset.InputPadding.Automatic = UserInfo['InputPadding'].Automatic
        Dataset.InputPadding.HardDimensions = list( np.array(UserInfo['InputPadding'].HardDimensions)[ Dataset.slicingInfo.slicingOrder ] )


        return Dataset

    def func_ModelParams():

        HardParams = WhichExperiment.HardParams
        def ReferenceForCascadeMethod(ModelIdea):

            _ , fullIndexes, _ = smallFuncs.NucleiSelection(ind=1)
            referenceLabel = {}

            if ModelIdea == 'HCascade':

                Name, Indexes = {}, {}
                for i in [1.1, 1.2, 1.3, 1.4]:
                    Name[i], Indexes[i], _ = smallFuncs.NucleiSelection(ind=i)

                for ixf in tuple(fullIndexes) + tuple([1.1, 1.2, 1.3, 1.4]):

                    if ixf in Indexes[1.1]: referenceLabel[ixf] = Name[1.1]
                    elif ixf in Indexes[1.2]: referenceLabel[ixf] = Name[1.2]
                    elif ixf in Indexes[1.3]: referenceLabel[ixf] = Name[1.3]
                    elif ixf in Indexes[1.4]: referenceLabel[ixf] = Name[1.4]
                    elif ixf == 1: referenceLabel[ixf] = 'None'
                    else: referenceLabel[ixf] = '1-THALAMUS'


            elif ModelIdea == 'Cascade':
                for ix in fullIndexes: referenceLabel[ix] = '1-THALAMUS' if ix != 1 else 'None'

            else:
                for ix in fullIndexes: referenceLabel[ix] = 'None'

            return referenceLabel

        def func_NumClasses():

            num_classes = len(UserInfo['simulation'].nucleus_Index) if HardParams.Model.MultiClass.Mode else 1
            if HardParams.Model.Method.havingBackGround_AsExtraDimension: num_classes += 1 
                
            return num_classes


        def fixing_NetworkParams_BasedOn_InputDim(dim):
            class kernel_size: 
                conv          = tuple([3]*dim)
                convTranspose = tuple([2]*dim)
                output        = tuple([1]*dim)

            class maxPooling: 
                strides   = tuple([2]*dim)
                pool_size = tuple([2]*dim)


            return kernel_size, maxPooling

        def func_Layer_Params(UserInfo):

            Layer_Params = HardParams.Model.Layer_Params
            
            kernel_size, maxPooling = fixing_NetworkParams_BasedOn_InputDim(UserInfo['simulation'].InputImage2Dvs3D)

            Layer_Params.FirstLayer_FeatureMap_Num = UserInfo['simulation'].FirstLayer_FeatureMap_Num
            Layer_Params.ConvLayer.Kernel_size = kernel_size()
            Layer_Params.MaxPooling = maxPooling()
            Layer_Params.Dropout.Value = UserInfo['DropoutValue']
            Layer_Params.class_weight.Mode = UserInfo['simulation'].Weighted_Class_Mode

            return Layer_Params

        HardParams.Template = UserInfo['Template']()
        HardParams.Machine.GPU_Index = str(UserInfo['simulation'].GPU_Index)

     
        HardParams.Model.metrics, _    = Metrics.MetricInfo(UserInfo['MetricIx'])
        HardParams.Model.optimizer, _  = Optimizers.OptimizerInfo(1, UserInfo['simulation'].Learning_Rate)
        HardParams.Model.num_Layers    = UserInfo['simulation'].num_Layers
        HardParams.Model.batch_size    = UserInfo['simulation'].batch_size
        HardParams.Model.epochs        = UserInfo['simulation'].epochs
        HardParams.Model.verbose       = UserInfo['simulation'].verbose
        HardParams.Model.DataGenerator = UserInfo['dataGenerator']()                
        HardParams.Model.Initialize    = UserInfo['InitializeB']
        HardParams.Model.architectureType = UserInfo['architectureType'] 
        HardParams.Model.Upsample      = UserInfo['upsample']()


        HardParams.Model.loss, _ = LossFunction.LossInfo(UserInfo['lossFunction_Index'] ) 

        HardParams.Model.Method.Type                  = UserInfo['Model_Method']
        HardParams.Model.Method.save_Best_Epoch_Model = UserInfo['simulation'].save_Best_Epoch_Model   
        HardParams.Model.Method.InputImage2Dvs3D      = UserInfo['simulation'].InputImage2Dvs3D
        HardParams.Model.Method.havingBackGround_AsExtraDimension = UserInfo['havingBackGround_AsExtraDimension']
        HardParams.Model.Method.Multiply_By_Thalmaus              = UserInfo['simulation'].Multiply_By_Thalmaus
        HardParams.Model.Method.Multiply_By_Rest_For_AV           = UserInfo['simulation'].Multiply_By_Rest_For_AV

        HardParams.Model.Method.Use_Coronal_Thalamus_InSagittal   = UserInfo['simulation'].Use_Coronal_Thalamus_InSagittal
        HardParams.Model.Method.Use_TestCases_For_Validation      = UserInfo['simulation'].Use_TestCases_For_Validation
        HardParams.Model.Method.ImClosePrediction                 = UserInfo['simulation'].ImClosePrediction

        HardParams.Model.MultiClass.Mode = UserInfo['simulation'].Multi_Class_Mode
        HardParams.Model.MultiClass.num_classes = func_NumClasses()
        HardParams.Model.Layer_Params = func_Layer_Params(UserInfo)

        if UserInfo['simulation'].nucleus_Index == 'all': 
            _, nucleus_Index,_ = smallFuncs.NucleiSelection(ind = 1)
        else:
            nucleus_Index = UserInfo['simulation'].nucleus_Index if isinstance(UserInfo['simulation'].nucleus_Index,list) else [UserInfo['simulation'].nucleus_Index]

        # AAA = ReferenceForCascadeMethod(HardParams.Model.Method.Type)
        # HardParams.Model.Method.ReferenceMask = AAA[nucleus_Index[0]]

        HardParams.Model.Method.ReferenceMask = ReferenceForCascadeMethod(HardParams.Model.Method.Type)[nucleus_Index[0]]
        HardParams.Model.Transfer_Learning = UserInfo['Transfer_Learning']

        return HardParams

    def ReadInputDimensions_NLayers(TrainModel_Address):
        with open(TrainModel_Address + '/UserInfo.json','rb') as f:   
            UserInfo_Load = json.load(f)            
        return UserInfo_Load['InputPadding_Dims'], UserInfo_Load['num_Layers']
        
    experiment, subExperiment = func_Exp_subExp_Names(UserInfo)  

    WhichExperiment.Experiment    = experiment
    WhichExperiment.SubExperiment = subExperiment
    WhichExperiment.address       = UserInfo['Experiments_Address']         
    WhichExperiment.HardParams    = func_ModelParams()
    WhichExperiment.Nucleus       = func_Nucleus(WhichExperiment.HardParams.Model.MultiClass.Mode)
    WhichExperiment.Dataset       = func_Dataset()
        
    # if UserInfo['simulation'].TestOnly: 
    #     InputDimensions, num_Layers = ReadInputDimensions_NLayers(experiment.address + '/models/' + subExperiment.name + '/' + WhichExperiment.Nucleus.name + '/sd' + str(WhichExperiment.Dataset.slicingInfo.slicingDim) )
    #     WhichExperiment.HardParams.Model.InputDimensions = InputDimensions
    #     WhichExperiment.HardParams.Model.num_Layers = num_Layers

    return WhichExperiment
    
def func_preprocess(UserInfo):

    def preprocess_Class():

        class normalize:
            Mode = True
            Method = 'MinMax'
            per_Subject = True
            per_Dataset = False


        class cropping:
            Mode = True
            Method = 'python'

        class biasCorrection:
            Mode = ''

        # TODO fix the justfornow
        class debug:
            doDebug = True
            PProcessExist = False  # rename it to preprocess exist
            justForNow = True # it checks the intermediate steps and if it existed don't reproduce it

        class preprocess:
            Mode = ''
            TestOnly = ''
            Debug = debug()
            # Augment = augment()
            Cropping = cropping()
            Normalize = normalize()
            BiasCorrection = biasCorrection()

        return preprocess()
    preprocess = preprocess_Class()

    preprocess.Mode                = UserInfo['preprocess'].Mode
    preprocess.BiasCorrection.Mode = UserInfo['preprocess'].BiasCorrection
    preprocess.Normalize           = UserInfo['normalize']()
    preprocess.TestOnly            = UserInfo['simulation'].TestOnly
    return preprocess

def func_Augment(UserInfo):

    def Augment_Class():
        class rotation:
            Mode = False
            AngleMax = 6

        class shift:
            Mode = False
            ShiftMax = 10

        class shear:
            Mode = False
            ShearMax = 0

        class linearAug:
            Mode = True
            Length = 8
            Rotation = rotation()
            Shift = shift()
            Shear = shear()

        class nonlinearAug:
            Mode = False
            Length = 2
        class augment:
            Mode = False
            Linear = linearAug()
            NonLinear = nonlinearAug()

        return augment()
    Augment = Augment_Class()

    Augment.Mode            = UserInfo['AugmentMode']
    Augment.Linear.Rotation = UserInfo['Augment_Rotation']()
    Augment.Linear.Shear    = UserInfo['Augment_Shear']()
    Augment.Linear.Length   = UserInfo['Augment_Linear_Length']
    Augment.NonLinear.Mode  = UserInfo['Augment_NonLinearMode']
    return Augment
    


