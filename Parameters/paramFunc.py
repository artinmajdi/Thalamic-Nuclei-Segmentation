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

def Run(UserInfo):

    class params:
        WhichExperiment, preprocess, Augment, directories = Classes()
        UserInfo = ''

    WhichExperiment, preprocess, Augment, directories = Classes()

    WhichExperiment.address = UserInfo['Experiments_Address']

    WhichExperiment.HardParams.Template = UserInfo['Template']()
    WhichExperiment.HardParams.Model.Method.Type  = UserInfo['Model_Method']
    WhichExperiment.HardParams.Model.metrics, _   = Metrics.MetricInfo(UserInfo['MetricIx'])
    WhichExperiment.HardParams.Model.optimizer, _ = Optimizers.OptimizerInfo(UserInfo['OptimizerIx'], UserInfo['simulation'].Learning_Rate)
    WhichExperiment.HardParams.Model.num_Layers   = UserInfo['simulation'].num_Layers
    WhichExperiment.HardParams.Model.batch_size   = UserInfo['simulation'].batch_size
    WhichExperiment.HardParams.Model.epochs       = UserInfo['simulation'].epochs
    WhichExperiment.HardParams.Model.InitializeFromThalamus = UserInfo['simulation'].Initialize_FromThalamus
    WhichExperiment.HardParams.Model.InitializeFromOlderModel = UserInfo['simulation'].Initialize_FromOlderModel


    WhichExperiment.HardParams.Machine.GPU_Index = str(UserInfo['simulation'].GPU_Index)
    print('GPU_Index:  ',WhichExperiment.HardParams.Machine.GPU_Index)

    if WhichExperiment.HardParams.Model.InitializeFromThalamus and WhichExperiment.HardParams.Model.InitializeFromOlderModel:
        print('WARNING:   initilization can only happen from one source')
        WhichExperiment.HardParams.Model.InitializeFromThalamus = False
        WhichExperiment.HardParams.Model.InitializeFromOlderModel = False

    WhichExperiment.Dataset.gapDilation = UserInfo['gapDilation']
    # WhichExperiment.Dataset.name, WhichExperiment.Dataset.address = datasets.DatasetsInfo(UserInfo['DatasetIx'])
    WhichExperiment.Dataset.HDf5.mode_saveTrue_LoadFalse = UserInfo['mode_saveTrue_LoadFalse']

    WhichExperiment.Dataset.slicingInfo = slicingInfoFunc(WhichExperiment.Dataset.slicingInfo, UserInfo['simulation'].slicingDim)

    WhichExperiment.SubExperiment.index = UserInfo['SubExperiment'].Index
    WhichExperiment.Experiment.index   = UserInfo['Experiments'].Index
    WhichExperiment.Experiment.tag     = UserInfo['Experiments'].Tag
    WhichExperiment.Experiment.name    = 'exp' + str(UserInfo['Experiments'].Index) + '_' + WhichExperiment.Experiment.tag if WhichExperiment.Experiment.tag else 'exp' + str(WhichExperiment.Experiment.index)
    WhichExperiment.Experiment.address = smallFuncs.mkDir(WhichExperiment.address + '/' + WhichExperiment.Experiment.name)
    # _, B = LossFunction.LossInfo(UserInfo['lossFunctionIx'])

    WhichExperiment.Dataset.InputPadding = UserInfo['InputPadding']
    WhichExperiment.Dataset.ReadTrain  = UserInfo['ReadTrain']()

    WhichExperiment = subExperimentName(UserInfo, WhichExperiment)

    # TODO I need to fix this to count for multiple nuclei
    WhichExperiment.Nucleus.Index = UserInfo['simulation'].nucleus_Index if isinstance(UserInfo['simulation'].nucleus_Index,list) else [UserInfo['simulation'].nucleus_Index]
    print('nucleus_Index', WhichExperiment.Nucleus.Index)

    WhichExperiment.Nucleus.name_Thalamus, WhichExperiment.Nucleus.FullIndexes, _ = smallFuncs.NucleiSelection( 1 )
    if len(WhichExperiment.Nucleus.Index) == 1 or not WhichExperiment.HardParams.Model.MultiClass.mode:
        WhichExperiment.Nucleus.name , _, _ = smallFuncs.NucleiSelection( WhichExperiment.Nucleus.Index[0] )
    else:
        WhichExperiment.Nucleus.name = ('MultiClass_' + str(WhichExperiment.Nucleus.Index)).replace(', ','_').replace('[','').replace(']','')

    WhichExperiment.HardParams.Model.Method.havingBackGround_AsExtraDimension = UserInfo['havingBackGround_AsExtraDimension']
    if UserInfo['havingBackGround_AsExtraDimension']:
        WhichExperiment.HardParams.Model.MultiClass.num_classes = len(WhichExperiment.Nucleus.Index) + 1 if WhichExperiment.HardParams.Model.MultiClass.mode else 2
    else:
        WhichExperiment.HardParams.Model.MultiClass.num_classes = len(WhichExperiment.Nucleus.Index) if WhichExperiment.HardParams.Model.MultiClass.mode else 1




    directories = smallFuncs.search_ExperimentDirectory(WhichExperiment)



    preprocess.Mode                = UserInfo['preprocess'].Mode
    preprocess.BiasCorrection.Mode = UserInfo['preprocess'].BiasCorrection
    preprocess.Normalize.Mode      = UserInfo['preprocess'].Normalize
    preprocess.Normalize.Method    = UserInfo['simulation'].NormalizaeMethod
    preprocess.TestOnly            = UserInfo['simulation'].TestOnly
    preprocess.Cropping            = UserInfo['cropping']

    Augment.Mode            = UserInfo['AugmentMode']
    Augment.Linear.Rotation = UserInfo['Augment_Rotation']()
    Augment.Linear.Shear    = UserInfo['Augment_Shear']()
    Augment.NonLinear.Mode  = UserInfo['Augment_NonLinearMode']

    WhichExperiment.HardParams.Model.Dropout.Value = UserInfo['DropoutValue']

    AAA = ReferenceForCascadeMethod(WhichExperiment.HardParams.Model.Method.Type)
    WhichExperiment.HardParams.Model.Method.ReferenceMask = AAA[WhichExperiment.Nucleus.Index[0]]

    WhichExperiment.HardParams.Model.Transfer_Learning = UserInfo['Transfer_Learning']()

    params.WhichExperiment = WhichExperiment
    params.preprocess      = preprocess
    params.directories     = directories
    params.UserInfo        = UserInfo
    params.Augment         = Augment


    if preprocess.TestOnly:
        # hist_params = pd.read_csv(directories.Train.Model + '/hist_params.csv').head()
        with open(directories.Train.Model + '/hist_params.pkl','rb') as f:
            hist_params = pickle.load(f)

        params.WhichExperiment.HardParams.Model.InputDimensions = [hist_params['InputDimensionsX'], hist_params['InputDimensionsY'],0]
        params.WhichExperiment.HardParams.Model.num_Layers = hist_params['num_Layers']

    return params

def slicingInfoFunc(slicingInfo, slicingDim):
    print('slicingDim:  ',slicingDim[0])
    slicingInfo.slicingDim = slicingDim[0]
    if slicingDim[0] == 0:
        slicingInfo.slicingOrder         = [1,2,0]
        slicingInfo.slicingOrder_Reverse = [2,0,1]
    elif slicingDim[0] == 1:
        slicingInfo.slicingOrder         = [2,0,1]
        slicingInfo.slicingOrder_Reverse = [1,2,0]
    else:
        slicingInfo.slicingOrder         = [0,1,2]
        slicingInfo.slicingOrder_Reverse = [0,1,2]

    return slicingInfo

def subExperimentName(UserInfo, WhichExperiment):

    WhichExperiment.SubExperiment.tag = UserInfo['SubExperiment'].Tag

    readAugmentTag = ''
    if UserInfo['Augment_Rotation'].Mode: readAugmentTag = 'wRot'   + str(UserInfo['Augment_Rotation'].AngleMax)
    elif UserInfo['Augment_Shear'].Mode:  readAugmentTag = 'wShear' + str(UserInfo['Augment_Shear'].ShearMax)

    if readAugmentTag: WhichExperiment.SubExperiment.tag += readAugmentTag

    # if int(UserInfo['simulation'].slicingDim[0]) != 2:
    WhichExperiment.SubExperiment.tag += '_sd' + str(UserInfo['simulation'].slicingDim[0])
    WhichExperiment.SubExperiment.tag += '_Dt' + str(UserInfo['DropoutValue'])
    if UserInfo['ReadTrain'].SRI: WhichExperiment.SubExperiment.tag += '_SRI'
    WhichExperiment.SubExperiment.name = 'sE' + str(WhichExperiment.SubExperiment.index) +  '_' + WhichExperiment.SubExperiment.tag

    # readAugmentTag = 'sE6_CascadewRot7_4cnts_sd2_Dt0.3'

    WhichExperiment.Dataset.ReadTrain.ReadAugments.Tag = readAugmentTag
    return WhichExperiment

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

def Classes():

    def DirectoriesFunc():
        class input:
            address , Subjects = '', {}
        class train:
            address , Model, Model_Thalamus, Input   = '' , '' , '' , input()

        class test:
            address, Result, Input = '' , '', input()

        class Directories:
            Train, Test = train(), test()

        return Directories()
    Directories = DirectoriesFunc()

    def WhichExperimentFunc():

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


                class method:
                    Type = ''
                    InitializeFromReference = True # from 3T or WMn for CSFn
                    ReferenceMask = ''
                    havingBackGround_AsExtraDimension = True

                return dropout, kernel_size, activation, convLayer, multiclass, maxPooling, method

            dropout, kernel_size, activation, convLayer, multiclass, maxPooling, method = ArchtiectureParams()

            class transfer_Learning:
                Mode = False
                Stage = 0 # 1
                FrozenLayers = [0,1]

            class model:
                architectureType = 'U-Net'
                epochs = ''
                batch_size = ''
                loss = ''
                metrics = ''
                optimizer = ''  # adamax Nadam Adadelta Adagrad  optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                num_Layers = ''
                InputDimensions = ''
                batchNormalization = True # True
                ConvLayer = convLayer()
                MaxPooling = maxPooling()
                Dropout = dropout()
                Activitation = activation()
                showHistory = True
                LabelMaxValue = 1
                class_weight = {}
                Measure_Dice_on_Train_Data = False
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
    WhichExperiment = WhichExperimentFunc()

    def AugmentFunc():
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
    augment = AugmentFunc()

    def preprocessFunc():

        class normalize:
            Mode = True
            Method = 'MinMax'


        class cropping:
            Mode = ''
            Method = ''

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
    preprocess = preprocessFunc()

    return WhichExperiment, preprocess, augment, Directories

    # class trainCase:
    #     def __init__(self, Image, Mask):
    #         self.Image = Image
    #         self.Mask  = Mask
