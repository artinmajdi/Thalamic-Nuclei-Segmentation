import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import modelFuncs.LossFunction as LossFunction
import modelFuncs.Metrics as Metrics
import modelFuncs.Optimizers as Optimizers
# from Parameters import Classes
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets

from copy import deepcopy
import pandas as pd
          
def Run(UserInfo):
    
    class params:
        WhichExperiment, preprocess, Augment, directories = Classes()
        UserInfo = ''

    WhichExperiment, preprocess, Augment, directories = Classes()

    WhichExperiment.address = smallFuncs.mkDir(UserInfo['Experiments_Address'])

    WhichExperiment.HardParams.Template.Image = UserInfo['Tempalte_Image']
    WhichExperiment.HardParams.Template.Mask  = UserInfo['Tempalte_Mask']
    WhichExperiment.HardParams.Model.MultiClass.mode = UserInfo['MultiClass_mode']
    WhichExperiment.HardParams.Model.loss, _      = LossFunction.LossInfo(UserInfo['lossFunctionIx'])
    WhichExperiment.HardParams.Model.Method.Type  = UserInfo['Model_Method']
    WhichExperiment.HardParams.Model.metrics, _   = Metrics.MetricInfo(UserInfo['MetricIx'])
    WhichExperiment.HardParams.Model.optimizer, _ = Optimizers.OptimizerInfo(UserInfo['OptimizerIx'], UserInfo['Learning_Rate'])
    WhichExperiment.HardParams.Model.num_Layers   = UserInfo['num_Layers']
    WhichExperiment.HardParams.Model.batch_size   = UserInfo['batch_size']
    WhichExperiment.HardParams.Model.epochs       = UserInfo['epochs']
    WhichExperiment.HardParams.Model.InitializeFromThalamus = UserInfo['Initialize_FromThalamus']
    WhichExperiment.HardParams.Model.InitializeFromOlderModel = UserInfo['Initialize_FromOlderModel']
    WhichExperiment.HardParams.Model.class_weight = UserInfo['class_weights']


    WhichExperiment.HardParams.Machine.GPU_Index = str(UserInfo['GPU_Index'])
    print('---------',WhichExperiment.HardParams.Machine.GPU_Index)

    if WhichExperiment.HardParams.Model.InitializeFromThalamus and WhichExperiment.HardParams.Model.InitializeFromOlderModel:
        print('WARNING:   initilization can only happen from one source')
        WhichExperiment.HardParams.Model.InitializeFromThalamus = False
        WhichExperiment.HardParams.Model.InitializeFromOlderModel = False

    WhichExperiment.Dataset.gapDilation = UserInfo['gapDilation']
    WhichExperiment.Dataset.name, WhichExperiment.Dataset.address = datasets.DatasetsInfo(UserInfo['DatasetIx'])
    WhichExperiment.Dataset.HDf5.mode_saveTrue_LoadFalse = UserInfo['mode_saveTrue_LoadFalse']

    # orderDim =       2: [0,1,2]
    # orderDim =       1: [2,0,1]
    # orderDim =       0: [1,2,0]
    print('---',UserInfo['slicingDim'])
    WhichExperiment.Dataset.slicingInfo.slicingDim = UserInfo['slicingDim'][0]
    if UserInfo['slicingDim'][0] == 0:
        WhichExperiment.Dataset.slicingInfo.slicingOrder         = [1,2,0]
        WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse = [2,0,1]
    elif UserInfo['slicingDim'][0] == 1:
        WhichExperiment.Dataset.slicingInfo.slicingOrder         = [2,0,1]
        WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse = [1,2,0]
    else:
        WhichExperiment.Dataset.slicingInfo.slicingOrder         = [0,1,2]
        WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse = [0,1,2]

    WhichExperiment.SubExperiment.index = UserInfo['SubExperiment_Index']
    WhichExperiment.Experiment.index = UserInfo['Experiments_Index']

    # if UserInfo['DatasetIx'] == 4:
    #     Experiments_Tag = '7T'
    # elif UserInfo['DatasetIx'] == 1:
    #     Experiments_Tag = 'SRI'
    # elif UserInfo['DatasetIx'] == 2:
    #     Experiments_Tag = 'Cropping'

    Experiments_Tag = UserInfo['Experiments_Tag']

    WhichExperiment.Experiment.tag = Experiments_Tag   # UserInfo['Experiments_Tag']
    WhichExperiment.Experiment.name = 'exp' + str(UserInfo['Experiments_Index']) + '_' + WhichExperiment.Experiment.tag if WhichExperiment.Experiment.tag else 'exp' + str(WhichExperiment.Experiment.index)
    WhichExperiment.Experiment.address = smallFuncs.mkDir(WhichExperiment.address + '/' + WhichExperiment.Experiment.name)
    # _, B = LossFunction.LossInfo(UserInfo['lossFunctionIx'])

    readAugmentTag, WhichExperiment = subExperimentName(UserInfo, WhichExperiment)

    # TODO I need to fix this to count for multiple nuclei
    WhichExperiment.Nucleus.Index = UserInfo['nucleus_Index'] if isinstance(UserInfo['nucleus_Index'],list) else [UserInfo['nucleus_Index']]
    
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

    WhichExperiment.Dataset.InputPadding.Automatic = UserInfo['InputPadding_Automatic']
    WhichExperiment.Dataset.InputPadding.HardDimensions = UserInfo['InputPadding_HardDimensions']
    WhichExperiment.Dataset.ReadAugments.Mode = UserInfo['readAugments']
    WhichExperiment.Dataset.ReadAugments.Tag = readAugmentTag

    WhichExperiment.Dataset.Read3T.Mode = UserInfo['read3T_Mode']
    WhichExperiment.Dataset.Read3T.Tag = UserInfo['read3T_Tag']

    if WhichExperiment.Dataset.InputPadding.Automatic:
        UserInfo['InputPadding_Automatic']
    directories = smallFuncs.search_ExperimentDirectory(WhichExperiment)
    if not Augment.Mode:  Augment.AugmentLength = 0
    preprocess.Cropping.Method = UserInfo['cropping_method']


    preprocess.Mode                = UserInfo['preprocessMode']
    preprocess.BiasCorrection.Mode = UserInfo['BiasCorrection']
    preprocess.Cropping.Mode       = UserInfo['Cropping']
    preprocess.Normalize.Mode      = UserInfo['Normalize']
    preprocess.Normalize.Method    = UserInfo['NormalizaeMethod']
    preprocess.TestOnly            = UserInfo['TestOnly']

    Augment.Mode                     = UserInfo['AugmentMode']
    Augment.Linear.Rotation.Mode     = UserInfo['Augment_Rotation']
    Augment.Linear.Rotation.AngleMax = UserInfo['Augment_AngleMax']

    Augment.Linear.Shear.Mode     = UserInfo['Augment_Shear']
    Augment.Linear.Shear.ShearMax = UserInfo['Augment_ShearMax']

    Augment.Linear.Shift.Mode        = UserInfo['Augment_Shift']
    Augment.Linear.Shift.ShiftMax    = UserInfo['Augment_ShiftMax']
    Augment.NonLinear.Mode           = UserInfo['Augment_NonLinearMode']

    WhichExperiment.HardParams.Model.Dropout.Value = UserInfo['dropout']

    AAA = ReferenceForCascadeMethod(WhichExperiment.HardParams.Model.Method.Type)
    WhichExperiment.HardParams.Model.Method.ReferenceMask = AAA[WhichExperiment.Nucleus.Index[0]]

    WhichExperiment.HardParams.Model.Transfer_Learning.Mode         = UserInfo['Transfer_Learning_Mode']
    WhichExperiment.HardParams.Model.Transfer_Learning.FrozenLayers = UserInfo['Transfer_Learning_Layers']

    params.WhichExperiment = WhichExperiment
    params.preprocess      = preprocess
    params.directories     = directories
    params.UserInfo        = UserInfo
    params.Augment         = Augment


    if preprocess.TestOnly:
        hist_params = pd.read_csv(directories.Train.Model + '/hist_params.csv').head()

        params.WhichExperiment.HardParams.Model.InputDimensions = [hist_params['InputDimensionsX'][0], hist_params['InputDimensionsY'][0],0]
        params.WhichExperiment.HardParams.Model.num_Layers = hist_params['num_Layers'][0]

    return params

def subExperimentName(UserInfo, WhichExperiment):

    readAugmentTag = ''
    if UserInfo['Augment_Rotation']: readAugmentTag = 'wRot'   + str(UserInfo['Augment_AngleMax'])
    elif UserInfo['Augment_Shear']:  readAugmentTag = 'wShear' + str(UserInfo['Augment_ShearMax'])   
    elif UserInfo['Augment_Shift']:  readAugmentTag = 'wShift' + str(UserInfo['Augment_ShiftMax'])  
    elif UserInfo['Augment_Merge']:  readAugmentTag = 'wMerge'

    
    WhichExperiment.SubExperiment.tag = UserInfo['SubExperiment_Tag']
    
    if readAugmentTag: WhichExperiment.SubExperiment.tag += '_Aug_' + readAugmentTag
        
    if int(UserInfo['slicingDim'][0]) != 2:
        WhichExperiment.SubExperiment.tag += '_sd' + str(UserInfo['slicingDim'][0])

    WhichExperiment.SubExperiment.tag += '_DrpOt' + str(UserInfo['dropout'])

    WhichExperiment.SubExperiment.name = 'subExp' + str(WhichExperiment.SubExperiment.index) +  '_' + WhichExperiment.SubExperiment.tag 

    return readAugmentTag, WhichExperiment
    
def ReferenceForCascadeMethod(ModelIdea):

    _ , fullIndexes, _ = smallFuncs.NucleiSelection(ind=1)
    referenceLabel = {}  

    if ModelIdea == 'Hierarchical_Cascade':

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

    # --------------------------------- Model --------------------------------

    class input:
        address , Subjects = '', {}
    class train:
        address , Model, Model_Thalamus, Input   = '' , '' , '' , input()

    class test:
        address, Result, Input = '' , '', input()

    class Directories:
        Train, Test = train, test


    class template:
        Image = ''
        Mask  = ''

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
        mode = ''

    class maxPooling:
        strides = (2,2)
        pool_size = (2,2)


    class method:
        Type = ''
        InitializeFromReference = True # from 3T or WMn for CSFn
        ReferenceMask = ''
        havingBackGround_AsExtraDimension = True


    # method.Type
    # 1. Normal
    # 2. Cascade
    # 3. Hierarchical_Cascade    

    class transfer_Learning:
        Mode = False
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
        
    class machine:
        WhichMachine = 'server'
        GPU_Index = ''

    class image:
        # SlicingDirection = 'axial'.lower()
        SaveMode = 'nifti'.lower()

    class nucleus:
        Organ = 'THALAMUS' # 'Hippocampus
        name = ''
        name_Thalamus = ''
        FullIndexes = ''
        Index = ''


    class hardParams:
        Model    = model()
        Template = template()
        Machine  = machine()
        Image    = image()

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

    # --------------------------------- Dataset --------------------------------

    class validation:
        percentage = 0.1
        fromKeras = False

    class testDs:
        mode = 'percentage' # 'names'
        percentage = 0.3
        subjects = ''

    # TODO IMPORT TEST SUBJECTS NAMES AS A LIST
    if 'names' in testDs.mode: # import testDs.subjects
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

    class Read3TFn:
        Mode = False
        Tag = '' # SRI

    class readAugmentFn:
        Mode = False
        Tag = ''
        
    class dataset:
        name = ''
        address = ''
        # CreatingTheExperiment = False
        Validation = validation()
        Test = testDs()
        check_vimp_SubjectName = True
        randomFlag = False
        slicingInfo = slicingDirection()
        gapDilation = 5
        gapOnSlicingDimention = 2
        InputPadding = inputPadding()
        ReadAugments = readAugmentFn()
        Read3T = Read3TFn()
        HDf5 = hDF5


    class WhichExperiment:
        Experiment    = experiment()
        SubExperiment = subExperiment()
        address = ''
        Nucleus = nucleus()
        HardParams = hardParams()
        Dataset = dataset()



    # --------------------------------- Augmentation --------------------------------
    class rotation:
        Mode = ''
        AngleMax = 6

    class shift:
        Mode = ''
        ShiftMax = 10

    class shear:
        Mode = ''
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
        Mode = ''
        Linear = linearAug()
        NonLinear = nonlinearAug()
        # LinearMode = True
        # LinearAugmentLength = 3  # number
        # NonLinearAugmentLength = 2
        # Rotation = rotation
        # Shift = shift
        # NonRigidWarp = ''

    # --------------------------------- Preprocess --------------------------------
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
        Augment = augment()
        Cropping = cropping()
        Normalize = normalize()
        BiasCorrection = biasCorrection()


    return WhichExperiment, preprocess, augment, Directories

    # class trainCase:
    #     def __init__(self, Image, Mask):
    #         self.Image = Image
    #         self.Mask  = Mask
