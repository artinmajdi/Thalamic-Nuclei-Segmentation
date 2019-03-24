import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import modelFuncs.LossFunction as LossFunction
import modelFuncs.Metrics as Metrics
import modelFuncs.Optimizers as Optimizers
# from Parameters import Classes
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import pickle
from copy import deepcopy
import pandas as pd

def Run(UserInfoB):

    class params:
        WhichExperiment = func_WhichExperiment(UserInfoB)
        preprocess      = func_preprocess(UserInfoB)
        Augment         = func_Augment(UserInfoB) 
        directories     = smallFuncs.search_ExperimentDirectory(WhichExperiment)
        UserInfo        = UserInfoB

    return params

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
                    mode = False

                class maxPooling:
                    strides = (2,2)
                    pool_size = (2,2)

                class upsample:
                    Scale = 2
                    Mode = False

                class method:
                    Type = ''
                    InitializeFromReference = True # from 3T or WMn for CSFn
                    ReferenceMask = ''
                    havingBackGround_AsExtraDimension = True
                    InputImage2Dvs3D = 2

                return dropout, kernel_size, activation, convLayer, multiclass, maxPooling, method

            dropout, kernel_size, activation, convLayer, multiclass, maxPooling, method = ArchtiectureParams()

            class transfer_Learning:
                Mode = False
                Stage = 0 # 1
                FrozenLayers = [0,1]

            class layer_Params:
                FirstLayer_FeatureMap_Num = 64
                batchNormalization = True
                ConvLayer = convLayer()
                MaxPooling = maxPooling()
                Dropout = dropout()
                Activitation = activation()
                class_weight = {}

            class model:
                architectureType = 'U-Net'
                epochs = ''
                batch_size = ''
                loss = ''
                metrics = ''
                optimizer = ''  # adamax Nadam Adadelta Adagrad  optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                
                num_Layers = ''
                InputDimensions = ''
                Layer_Params = layer_Params()
                showHistory = True
                LabelMaxValue = 1                
                Measure_Dice_on_Train_Data = True
                MultiClass = multiclass()
                #! only one of these two can be true at the same time
                InitializeFromThalamus = ''
                InitializeFromOlderModel = ''
                Method = method()
                paddingErrorPatience = 20
                Transfer_Learning = transfer_Learning()
                ManualDataGenerator = False
                

            lossFunctionIx = 5
            model.loss, _ = LossFunction.LossInfo(lossFunctionIx)

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

        class subExperiment:
            index = ''
            tag = ''
            name = ''
            name_thalamus = ''

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

            class readTrain:
                Main = True
                ET = True
                SRI = True
                ReadAugments = readAugmentFn()

            class dataset:
                name = ''
                address = ''
                Validation = validation()
                Test = testDs()
                check_vimp_SubjectName = True
                randomFlag = False
                slicingInfo = slicingDirection()
                gapDilation = 5
                gapOnSlicingDimention = 2
                InputPadding = inputPadding()
                ReadTrain = readTrain()
                HDf5 = hDF5()

            Dataset_Index = 4
            dataset.name, dataset.address = datasets.DatasetsInfo(Dataset_Index)
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
                NucleusName = ('MultiClass_' + str(NucleusIndex)).replace(', ','_').replace('[','').replace(']','')

            return NucleusName

        nucleus_Index = UserInfo['simulation'].nucleus_Index if isinstance(UserInfo['simulation'].nucleus_Index,list) else [UserInfo['simulation'].nucleus_Index]
        class nucleus:
            Organ = 'THALAMUS'
            name = Experiment_Nucleus_Name_MClass(nucleus_Index , MultiClassMode )
            name_Thalamus, FullIndexes, _ = smallFuncs.NucleiSelection( 1 )
            Index = nucleus_Index

        return nucleus

    def func_Dataset(ReadAugmentsTag):
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

        Dataset.ReadTrain  = UserInfo['ReadTrain']()
        Dataset.ReadTrain.ReadAugments.Tag = ReadAugmentsTag

        Dataset.gapDilation = UserInfo['gapDilation']
        Dataset.HDf5.mode_saveTrue_LoadFalse = UserInfo['mode_saveTrue_LoadFalse']
        Dataset.slicingInfo = slicingInfoFunc()

        Dataset.InputPadding.Automatic = UserInfo['InputPadding'].Automatic
        Dataset.InputPadding.HardDimensions = UserInfo['InputPadding'].HardDimensions[ Dataset.slicingInfo.slicingOrder ]

        return Dataset

    def func_Experiment_SubExp():
        def subExperimentName(UserInfo):

            SubExperimentTag = UserInfo['SubExperiment'].Tag

            readAugmentTag = ''
            if UserInfo['Augment_Rotation'].Mode: readAugmentTag = 'wRot'   + str(UserInfo['Augment_Rotation'].AngleMax)
            elif UserInfo['Augment_Shear'].Mode:  readAugmentTag = 'wShear' + str(UserInfo['Augment_Shear'].ShearMax)

            if readAugmentTag: SubExperimentTag += readAugmentTag

            # if int(UserInfo['simulation'].slicingDim[0]) != 2:
            SubExperimentTag += '_sd' + str(UserInfo['simulation'].slicingDim[0])
            SubExperimentTag += '_Dt' + str(UserInfo['DropoutValue'])
            # SubExperimentTag += '_LR' + str(UserInfo['simulation'].Learning_Rate)

            if UserInfo['ReadTrain'].SRI: SubExperimentTag += '_SRI'    

            return SubExperimentTag, readAugmentTag

        class experiment:
            index = UserInfo['Experiments'].Index
            tag = UserInfo['Experiments'].Tag
            name = 'exp' + str(UserInfo['Experiments'].Index) + '_' + UserInfo['Experiments'].Tag if UserInfo['Experiments'].Tag else 'exp' + str(UserInfo['Experiments'].Index)
            address = smallFuncs.mkDir(UserInfo['Experiments_Address'] + '/' + name)
 
        SubExperimentTag, ReadAugments = subExperimentName(UserInfo)  
        class subExperiment:
            index = UserInfo['SubExperiment'].Index
            tag = SubExperimentTag
            name = 'sE' + str(UserInfo['SubExperiment'].Index) +  '_' + SubExperimentTag
            name_thalamus = ''

        return experiment, subExperiment, ReadAugments

    def func_ModelParams():

        HardParams = WhichExperiment.HardParams
        def ReferenceForCascadeMethod(ModelIdea):

            _ , fullIndexes, _ = smallFuncs.NucleiSelection(ind=1)
            referenceLabel = {}

            if ModelIdea == 'HCascade':

                Name, Indexes = {}, {}
                for i in [1.1, 1.2, 1.3]:
                    Name[i], Indexes[i], _ = smallFuncs.NucleiSelection(ind=i)

                for ixf in tuple(fullIndexes) + tuple([1.1, 1.2, 1.3]):

                    if ixf in Indexes[1.1]: referenceLabel[ixf] = Name[1.1]
                    elif ixf in Indexes[1.2]: referenceLabel[ixf] = Name[1.2]
                    elif ixf in Indexes[1.3]: referenceLabel[ixf] = Name[1.3]
                    elif ixf == 1: referenceLabel[ixf] = 'None'
                    else: referenceLabel[ixf] = '1-THALAMUS'


            elif ModelIdea == 'Cascade':
                for ix in fullIndexes: referenceLabel[ix] = '1-THALAMUS' if ix != 1 else 'None'

            else:
                for ix in fullIndexes: referenceLabel[ix] = 'None'

            return referenceLabel

        def func_NumClasses():

            num_classes = len(nucleus_Index) if HardParams.Model.MultiClass.mode else 1
            if HardParams.Model.Method.havingBackGround_AsExtraDimension: num_classes += 1 
                
            return num_classes

        def func_Initialize(UserInfo):
            Initialize_From_Thalamus, Initialize_From_OlderModel = UserInfo['simulation'].Initialize_FromThalamus , UserInfo['simulation'].Initialize_FromOlderModel
            if Initialize_From_Thalamus and Initialize_From_OlderModel:
                print('WARNING:   initilization can only happen from one source')
                Initialize_From_Thalamus, Initialize_From_OlderModel = False , False

            return Initialize_From_Thalamus, Initialize_From_OlderModel

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
            return Layer_Params

        HardParams.Template = UserInfo['Template']()
        HardParams.Machine.GPU_Index = str(UserInfo['simulation'].GPU_Index)

        HardParams.Model.Method.Type  = UserInfo['Model_Method']
        HardParams.Model.metrics, _   = Metrics.MetricInfo(UserInfo['MetricIx'])
        HardParams.Model.optimizer, _ = Optimizers.OptimizerInfo(1, UserInfo['simulation'].Learning_Rate)
        HardParams.Model.num_Layers   = UserInfo['simulation'].num_Layers
        HardParams.Model.batch_size   = UserInfo['simulation'].batch_size
        HardParams.Model.epochs       = UserInfo['simulation'].epochs

        Initialize_From_Thalamus, Initialize_From_OlderModel = func_Initialize(UserInfo)
        HardParams.Model.InitializeFromThalamus = Initialize_From_Thalamus
        HardParams.Model.InitializeFromOlderModel = Initialize_From_OlderModel

        HardParams.Model.Method.InputImage2Dvs3D = UserInfo['simulation'].InputImage2Dvs3D
        HardParams.Model.Method.havingBackGround_AsExtraDimension = UserInfo['havingBackGround_AsExtraDimension']

        HardParams.Model.MultiClass.num_classes = func_NumClasses()
        HardParams.Model.Layer_Params = func_Layer_Params(UserInfo)

        nucleus_Index = UserInfo['simulation'].nucleus_Index if isinstance(UserInfo['simulation'].nucleus_Index,list) else [UserInfo['simulation'].nucleus_Index]

        # AAA = ReferenceForCascadeMethod(HardParams.Model.Method.Type)
        # HardParams.Model.Method.ReferenceMask = AAA[nucleus_Index[0]]

        HardParams.Model.Method.ReferenceMask = ReferenceForCascadeMethod(HardParams.Model.Method.Type)[nucleus_Index[0]]
        HardParams.Model.Transfer_Learning = UserInfo['Transfer_Learning']()

        return HardParams

    def ReadInputDimensions_NLayers(TrainModel_Address):
        with open(TrainModel_Address + '/hist_params.pkl','rb') as f:   hist_params = pickle.load(f)

        InputDimensions = [hist_params['InputDimensionsX'], hist_params['InputDimensionsY'], hist_params['InputDimensionsZ']]
        num_Layers = hist_params['num_Layers']

        return InputDimensions, num_Layers
        
    experiment, subExperiment , ReadAugments = func_Experiment_SubExp()  

    WhichExperiment.Experiment    = experiment()
    WhichExperiment.SubExperiment = subExperiment()
    WhichExperiment.address       = UserInfo['Experiments_Address']         
    WhichExperiment.HardParams    = func_ModelParams()
    WhichExperiment.Nucleus       = func_Nucleus(WhichExperiment.HardParams.Model.MultiClass.mode)
    WhichExperiment.Dataset       = func_Dataset(ReadAugments)
        
    if UserInfo['simulation'].TestOnly: 
        InputDimensions, num_Layers = ReadInputDimensions_NLayers(experiment.address + '/models/' + subExperiment.name + '/' + WhichExperiment.Nucleus.name)
        WhichExperiment.HardParams.Model.InputDimensions
        WhichExperiment.HardParams.Model.num_Layers

    return WhichExperiment
    
def func_preprocess(UserInfo):

    def preprocess_Class():

        class normalize:
            Mode = True
            Method = 'MinMax'


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
    Augment.NonLinear.Mode  = UserInfo['Augment_NonLinearMode']
    return Augment
    
# def Directories_Class():
#     class input:
#         address , Subjects = '', {}
#     class train:
#         address , Model, Model_Thalamus, Input   = '' , '' , '' , input()

#     class test:
#         address, Result, Input = '' , '', input()

#     class Directories:
#         Train, Test = train(), test()

#     return Directories()

