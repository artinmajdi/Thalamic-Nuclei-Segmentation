import os
import numpy as np
import json
from modelFuncs import LossFunction, Metrics, Optimizers
from otherFuncs import smallFuncs


def Run(UserInfoB):
    """
    class normalize:           
        Mode = True
        Method = '1Std0Mean'  # MinMax  -  1Std0Mean  -  Both

    class Preprocess:
        Mode             = False
        BiasCorrection   = False
        Cropping         = True
        Reslicing        = True
        save_debug_files = True
        Normalize        = normalize()
    """

    class Params:
        WhichExperiment = func_WhichExperiment(UserInfoB)
        preprocess = UserInfoB['preprocess']()
        directories = smallFuncs.search_ExperimentDirectory(WhichExperiment)
        UserInfo = UserInfoB

    return Params()


def func_WhichExperiment(UserInfo):
    USim = UserInfo['simulation']

    # def func_Nucleus():
    #     # nucleus_Index = USim.nucleus_Index # if isinstance(USim.nucleus_Index, list) else [USim.nucleus_Index]

    #     class nucleus:
    #         Index = UserInfo['simulation'].nucleus_Index 
    #         name = '1-THALAMUS' if len(Index) == 1 else 'MultiClass_24567891011121314'
    #         name_Thalamus, FullIndexes, FullNames = smallFuncs.NucleiSelection(1)
            

    #     return nucleus

    def func_Dataset():

        def datasetFunc():
            class validation:
                percentage = 0.1
                fromKeras = False

            class testDs:
                mode = 'percentage'  # 'names'
                percentage = 0.3
                subjects = ''

            if 'names' in testDs.mode:
                testDs.subjects = list([''])

            class slicingDirection:
                slicingOrder = [0, 1, 2]
                slicingOrder_Reverse = [0, 1, 2]
                slicingDim = 2

            class inputPadding:
                Automatic = True
                HardDimensions = ''

            class hDF5:
                mode = False
                mode_saveTrue_LoadFalse = True

            class dataset:
                name = ''
                address = ''
                Validation = validation()
                Test = testDs()
                check_case_SubjectName = True
                randomFlag = True
                slicingInfo = slicingDirection()
                gapDilation = 5
                gapOnSlicingDimention = 2
                InputPadding = inputPadding()
                HDf5 = hDF5()

            return dataset

        Dataset = datasetFunc()

        def slicingInfoFunc():
            class slicingInfo:
                slicingOrder = ''
                slicingOrder_Reverse = ''
                slicingDim = USim.slicingDim[0]

            if slicingInfo.slicingDim == 0:
                slicingInfo.slicingOrder = [1, 2, 0]
                slicingInfo.slicingOrder_Reverse = [2, 0, 1]
            elif slicingInfo.slicingDim == 1:
                slicingInfo.slicingOrder = [2, 0, 1]
                slicingInfo.slicingOrder_Reverse = [1, 2, 0]
            else:
                slicingInfo.slicingOrder = [0, 1, 2]
                slicingInfo.slicingOrder_Reverse = [0, 1, 2]

            return slicingInfo

        Dataset.slicingInfo = slicingInfoFunc()
        Dataset.check_case_SubjectName = UserInfo['simulation'].check_case_SubjectName
        Dataset.InputPadding.Automatic = UserInfo['InputPadding']().Automatic
        Dataset.InputPadding.HardDimensions = list(
            np.array(UserInfo['InputPadding']().HardDimensions)[Dataset.slicingInfo.slicingOrder])
        return Dataset

    def func_ModelParams():

        def HardParamsFuncs():
            def ArchtiectureParams():
                class dropout:
                    Mode = True
                    Value = 0.3

                class kernel_size:
                    conv = (3, 3)
                    convTranspose = (2, 2)
                    output = (1, 1)

                class activation:
                    layers = 'relu'
                    output = 'sigmoid'

                class convLayer:
                    Kernel_size = kernel_size()
                    padding = 'SAME'  # valid

                class multiclass:
                    num_classes = ''
                    Mode = True

                class maxPooling:
                    strides = (2, 2)
                    pool_size = (2, 2)

                class method:
                    Type = ''
                    ReferenceMask = ''
                    InputImage2Dvs3D = 2
                    save_Best_Epoch_Model = True
                    Use_Coronal_Thalamus_InSagittal = True
                    Use_TestCases_For_Validation = True
                    ImClosePrediction = True

                return dropout, activation, convLayer, multiclass, maxPooling, method

            dropout, activation, convLayer, multiclass, maxPooling, method = ArchtiectureParams()

            class classWeight:
                Weight = {0: 1, 1: 1}
                Mode = True

            class layer_Params:
                FirstLayer_FeatureMap_Num = 20
                batchNormalization = True
                ConvLayer = convLayer()
                MaxPooling = maxPooling()
                Dropout = dropout()
                Activitation = activation()
                class_weight = classWeight()

            class InitializeB:
                mode = True
                init_address = ''

            class model:
                architectureType = 'Res_Unet2'
                epochs = 50
                batch_size = 50
                Learning_Rate = 1e-3
                LR_Scheduler = True
                loss = 7
                metrics = ''
                # e.g. optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                #      - adamax
                #      - Nadam
                #      - Adadelta
                #      - Adagrad
                optimizer = ''
                verbose = 2
                num_Layers = 3
                InputDimensions = ''
                Layer_Params = layer_Params()
                showHistory = True
                LabelMaxValue = 1
                Measure_Dice_on_Train_Data = False
                MultiClass = multiclass()
                Initialize = InitializeB()
                Method = method()
                paddingErrorPatience = 200

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
                Mask = ''

            class hardParams:
                Model = model
                Template = template()
                Machine = machine()
                Image = image()

            return hardParams

        HardParams = HardParamsFuncs()

        def ReferenceForCascadeMethod():

            _, fullIndexes, _ = smallFuncs.NucleiSelection(ind=1)
            referenceLabel = {}

            for ix in fullIndexes:
                referenceLabel[ix] = '1-THALAMUS' if ix != 1 else 'None'

            return referenceLabel

        # def func_NumClasses():
        #     return len(USim.nucleus_Index) + 1

        def fixing_NetworkParams_BasedOn_InputDim(dim):
            class kernel_size:
                conv = tuple([3] * dim)
                convTranspose = tuple([2] * dim)
                output = tuple([1] * dim)

            class maxPooling:
                strides = tuple([2] * dim)
                pool_size = tuple([2] * dim)

            return kernel_size, maxPooling

        def func_Layer_Params():

            Layer_Params = HardParams.Model.Layer_Params

            kernel_size, maxPooling = fixing_NetworkParams_BasedOn_InputDim(2)
            Layer_Params.ConvLayer.Kernel_size = kernel_size()
            Layer_Params.MaxPooling = maxPooling()

            return Layer_Params

        HardParams.Template = UserInfo['Template']
        HardParams.Machine.GPU_Index = str(USim.GPU_Index)
        HardParams.Model.metrics, _ = Metrics.MetricInfo(3)
        HardParams.Model.optimizer, _ = Optimizers.OptimizerInfo(1, USim.Learning_Rate)
        HardParams.Model.num_Layers = USim.num_Layers
        HardParams.Model.batch_size = USim.batch_size
        HardParams.Model.epochs = USim.epochs
        HardParams.Model.Learning_Rate = USim.Learning_Rate
        HardParams.Model.LR_Scheduler = USim.LR_Scheduler
        HardParams.Model.Initialize = UserInfo['initialize']()

        HardParams.Model.architectureType = USim.architectureType
        HardParams.Model.Layer_Params.FirstLayer_FeatureMap_Num = USim.FirstLayer_FeatureMap_Num

        HardParams.Model.loss, _ = LossFunction.LossInfo(USim.lossFunction_Index)
        HardParams.Model.Method.Use_TestCases_For_Validation = USim.Use_TestCases_For_Validation

        HardParams.Model.MultiClass.num_classes = len(USim.nucleus_Index) + 1
        HardParams.Model.Layer_Params = func_Layer_Params()

        # if USim.nucleus_Index == 'all':
        #     _, nucleus_Index, _ = smallFuncs.NucleiSelection(ind=1)
        # else:
        nucleus_Index = USim.nucleus_Index if isinstance(USim.nucleus_Index, list) else [USim.nucleus_Index]

        HardParams.Model.Method.ReferenceMask = ReferenceForCascadeMethod()[nucleus_Index[0]]

        return HardParams

    def ReadInputDimensions_NLayers(TrainModel_Address):
        with open(TrainModel_Address + '/UserInfo.json', 'rb') as f:
            UserInfo_Load = json.load(f)
        return UserInfo_Load['InputPadding_Dims'], UserInfo_Load['num_Layers']

    class nucleus:
        Index = USim.nucleus_Index 
        name = '1-THALAMUS' if len(Index) == 1 else 'MultiClass_24567891011121314'
        name_Thalamus, FullIndexes, FullNames = smallFuncs.NucleiSelection(1)

    class WhichExperiment:
        Experiment = UserInfo['experiment']()
        HardParams = func_ModelParams()
        Nucleus    = nucleus()
        Dataset    = func_Dataset()
        TestOnly   = USim.TestOnly

    WE = WhichExperiment.Experiment
    dir_input_dimension = WE.exp_address + '/' + WE.subexperiment_name + '/' + WhichExperiment.Nucleus.name + \
                          '/sd' + str(WhichExperiment.Dataset.slicingInfo.slicingDim)

    if UserInfo['simulation'].use_train_padding_size and USim.TestOnly._mode and os.path.isfile(
            dir_input_dimension + '/UserInfo.json'):
        InputDimensions, num_Layers = ReadInputDimensions_NLayers(dir_input_dimension)

        WhichExperiment.Dataset.InputPadding.Automatic = False
        WhichExperiment.Dataset.InputPadding.HardDimensions = InputDimensions
        WhichExperiment.HardParams.Model.InputDimensions = InputDimensions
        WhichExperiment.HardParams.Model.num_Layers = num_Layers

    return WhichExperiment
