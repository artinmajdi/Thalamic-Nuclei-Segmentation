import os
import sys
import numpy as np
import json
import pathlib


sys.path.append(str(pathlib.Path(__file__).parent.parent))

from modelFuncs import LossFunction, Metrics, Optimizers
from otherFuncs import smallFuncs


def Run(UserInfoB, terminal=True):
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

    # Updating the user info with terminal entries
    if terminal:
        UserInfoB = smallFuncs.terminalEntries(UserInfoB)

        
    class Params:
        WhichExperiment = func_WhichExperiment(UserInfoB)
        preprocess = UserInfoB['preprocess']()
        directories = smallFuncs.search_ExperimentDirectory(WhichExperiment)
        UserInfo = UserInfoB

    return Params()


def func_WhichExperiment(UserInfo):
    USim = UserInfo['simulation']

    def func_Nucleus(MultiClassMode):
        nucleus_Index = USim.nucleus_Index if isinstance(USim.nucleus_Index, list) else [USim.nucleus_Index]

        class nucleus:
            name = '1-THALAMUS' if len(nucleus_Index) == 1 else 'MultiClass_24567891011121314'
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
        Dataset.check_case_SubjectName = UserInfo['simulation'].check_case_SubjectName
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
        HardParams.Model.Initialize.init_address = pathlib.Path(UserInfo['initialize'].init_address)

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

    class WhichExperiment:
        Experiment = UserInfo['experiment']()
        HardParams = func_ModelParams()
        Nucleus = func_Nucleus(WhichExperiment.HardParams.Model.MultiClass.Mode)
        Dataset = func_Dataset()
        TestOnly = USim.TestOnly

    WE = WhichExperiment.Experiment
    dir_input_dimension = WE.exp_address + '/' + WE.subexperiment_name + '/' + WhichExperiment.Nucleus.name + '/sd' + str(
        WhichExperiment.Dataset.slicingInfo.slicingDim)

    if UserInfo['simulation'].use_train_padding_size and USim.TestOnly.mode and os.path.isfile(
            dir_input_dimension + '/UserInfo.json'):
        InputDimensions, num_Layers = ReadInputDimensions_NLayers(dir_input_dimension)

        WhichExperiment.Dataset.InputPadding.Automatic = False
        WhichExperiment.Dataset.InputPadding.HardDimensions = InputDimensions
        WhichExperiment.HardParams.Model.InputDimensions = InputDimensions
        WhichExperiment.HardParams.Model.num_Layers = num_Layers

    return WhichExperiment
