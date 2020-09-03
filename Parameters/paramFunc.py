import os
import sys
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import modelFuncs.LossFunction as LossFunction
import modelFuncs.Metrics as Metrics
import modelFuncs.Optimizers as Optimizers
import otherFuncs.smallFuncs as smallFuncs


def Run(UserInfoB, terminal=False):
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

    def WhichExperiment_Class():

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
                    havingBackGround_AsExtraDimension = True
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
                modality_default = 'wmn'  # 'wmn' 'csfn'
                init_address = '/array/ssd/msmajdi/code/Trained_Models/WMn/'

            class model:
                architectureType = 'Res_Unet2'
                epochs = 50
                batch_size = 50
                Learning_Rate = 1e-3
                LR_Scheduler = True
                loss = 7
                metrics = ''
                optimizer = ''  # adamax Nadam Adadelta Adagrad  optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
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

        hardParams = HardParamsFuncs()

        class experiment:
            exp_address = ''
            train_address = ''
            test_address = ''
            init_address = ''
            ReadAugments_Mode = False

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
                check_vimp_SubjectName = True
                randomFlag = True
                slicingInfo = slicingDirection()
                gapDilation = 5
                gapOnSlicingDimention = 2
                InputPadding = inputPadding()
                HDf5 = hDF5()

            return dataset

        dataset = datasetFunc()

        class nucleus:
            name = ''
            Index = ''
            FullIndexes = ''
            FullNames = ''

        class WhichExperiment:
            Experiment = experiment()
            Nucleus = nucleus()
            HardParams = hardParams()
            Dataset = dataset()

        return WhichExperiment()

    WhichExperiment = WhichExperiment_Class()

    def func_Nucleus(MultiClassMode):
        nucleus_Index = USim.nucleus_Index if isinstance(USim.nucleus_Index, list) else [USim.nucleus_Index]

        class nucleus:
            name = '1-THALAMUS' if len(nucleus_Index) == 1 else 'MultiClass_24567891011121314/'
            name_Thalamus, FullIndexes, FullNames = smallFuncs.NucleiSelection(1)
            Index = nucleus_Index

        return nucleus

    def func_Dataset():
        Dataset = WhichExperiment.Dataset

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
        Dataset.check_vimp_SubjectName = UserInfo['simulation'].check_vimp_SubjectName
        Dataset.InputPadding.Automatic = UserInfo['InputPadding']().Automatic
        Dataset.InputPadding.HardDimensions = list(
            np.array(UserInfo['InputPadding']().HardDimensions)[Dataset.slicingInfo.slicingOrder])
        return Dataset

    def func_ModelParams():

        HardParams = WhichExperiment.HardParams

        def ReferenceForCascadeMethod():

            _, fullIndexes, _ = smallFuncs.NucleiSelection(ind=1)
            referenceLabel = {}

            for ix in fullIndexes:
                referenceLabel[ix] = '1-THALAMUS' if ix != 1 else 'None'

            return referenceLabel

        def func_NumClasses():

            num_classes = len(USim.nucleus_Index) if HardParams.Model.MultiClass.Mode else 1
            num_classes += 1
            return num_classes

        def fixing_NetworkParams_BasedOn_InputDim(dim):
            class kernel_size:
                conv = tuple([3] * dim)
                convTranspose = tuple([2] * dim)
                output = tuple([1] * dim)

            class maxPooling:
                strides = tuple([2] * dim)
                pool_size = tuple([2] * dim)

            return kernel_size, maxPooling

        def func_Layer_Params(UserInfo):

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
        HardParams.Model.Learning_Rate = UserInfo['simulation'].Learning_Rate
        HardParams.Model.LR_Scheduler = UserInfo['simulation'].LR_Scheduler
        HardParams.Model.Initialize = UserInfo['initialize']()
        HardParams.Model.architectureType = UserInfo['simulation'].architectureType
        HardParams.Model.Layer_Params.FirstLayer_FeatureMap_Num = UserInfo['simulation'].FirstLayer_FeatureMap_Num

        HardParams.Model.loss, _ = LossFunction.LossInfo(USim.lossFunction_Index)
        HardParams.Model.Method.Use_TestCases_For_Validation = USim.Use_TestCases_For_Validation

        HardParams.Model.MultiClass.num_classes = func_NumClasses()
        HardParams.Model.Layer_Params = func_Layer_Params(UserInfo)

        if USim.nucleus_Index == 'all':
            _, nucleus_Index, _ = smallFuncs.NucleiSelection(ind=1)
        else:
            nucleus_Index = USim.nucleus_Index if isinstance(USim.nucleus_Index, list) else [USim.nucleus_Index]

        HardParams.Model.Method.ReferenceMask = ReferenceForCascadeMethod()[nucleus_Index[0]]

        return HardParams

    def ReadInputDimensions_NLayers(TrainModel_Address):
        with open(TrainModel_Address + '/UserInfo.json', 'rb') as f:
            UserInfo_Load = json.load(f)
        return UserInfo_Load['InputPadding_Dims'], UserInfo_Load['num_Layers']

    WhichExperiment.Experiment = UserInfo['experiment']()
    WhichExperiment.HardParams = func_ModelParams()
    WhichExperiment.Nucleus    = func_Nucleus(WhichExperiment.HardParams.Model.MultiClass.Mode)
    WhichExperiment.Dataset    = func_Dataset()
    WhichExperiment.TestOnly   = USim.TestOnly

    """
    def adding_TransferLearningParams(WhichExperiment):

        def params_bestResUnet2(slicing_orientation):
            if slicing_orientation == 0:
                FM, NL = 40, 3
            elif slicing_orientation == 1:
                FM, NL = 30, 3
            elif slicing_orientation == 2:
                FM, NL = 20, 3

            return FM, NL, 'Res_Unet2'

        class best_WMn_Model:
            def __init__(self, WhichExperiment):
                SD = WhichExperiment.Dataset.slicingInfo.slicingDim

                loss_function = 'MyLogDice'
                EXP_address = '/array/ssd/msmajdi/experiments/keras/exp6/models/'

                self.FM, self.NL, architectureType = params_bestResUnet2(SD)
                Tag = 'sE12_Cascade_FM' + str(self.FM) + '_' + architectureType + '_NL' + str(
                    self.NL) + '_LS_' + loss_function + '_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/'

                self.address = EXP_address + Tag + WhichExperiment.Nucleus.name + '/sd' + str(SD) + '/model.h5'

        return best_WMn_Model(WhichExperiment)

    WhichExperiment.HardParams.Model.Best_WMn_Model = adding_TransferLearningParams(WhichExperiment)
    """

    WE = WhichExperiment.Experiment
    dir_input_dimension = WE.exp_address + '/' + WE.subexperiment_name + '/' + WhichExperiment.Nucleus.name + '/sd' + str(
        WhichExperiment.Dataset.slicingInfo.slicingDim)

    if UserInfo['simulation'].use_train_padding_size and USim.TestOnly and os.path.isfile( dir_input_dimension + '/UserInfo.json' ):
        InputDimensions, num_Layers = ReadInputDimensions_NLayers(dir_input_dimension)

        WhichExperiment.Dataset.InputPadding.Automatic      = False
        WhichExperiment.Dataset.InputPadding.HardDimensions = InputDimensions
        WhichExperiment.HardParams.Model.InputDimensions    = InputDimensions
        WhichExperiment.HardParams.Model.num_Layers         = num_Layers

    return WhichExperiment

