
class experiment:
    exp_address   = '/array/ssd/msmajdi/experiments/exp6/'
    train_address = '/array/ssd/msmajdi/experiments/data/train/'
    test_address  = '/array/ssd/msmajdi/experiments/data/test/'
    subexperiment_name = 'test_02_wmn'
    ReadAugments_Mode  = True
    code_address  = '/array/ssd/msmajdi/code/'

""" if init_address will be left empty, the default address will be used for initialization """
class initialize:
    mode = True
    modality_default = 'wmn' # 'wmn' 'csfn'
    init_address  = '' # '/array/ssd/msmajdi/code/Trained_Models/WMn/'

Model_Method = 'Cascade'

class thalamic_side:
    left  = True
    right = True
    # active_side = ''  # can be left empty

class normalize:
    """ Method: 
        MinMax
        1Std0Mean
        Both       """
    Mode   = True
    Method = '1Std0Mean'
    
class preprocess:
    Mode             = True
    BiasCorrection   = False
    Cropping         = True
    Reslicing        = True
    save_debug_files = True
    Normalize        = normalize()


class simulation:
    def __init__(self):
        self.TestOnly      = False
        self.epochs        = 5
        self.GPU_Index     = "1"
        self.batch_size    = 10
        self.Use_TestCases_For_Validation = True
        self.ImClosePrediction =  True
        self.Multi_Class_Mode = True
        self.LR_Scheduler = True
        self.ReadAugments_Mode = True
        self.Learning_Rate = 1e-3
        self.num_Layers = 3
        self.lossFunction_Index = 7
        self.nucleus_Index = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.slicingDim = [2,1,0]
        self.use_train_padding_size = False
        self.check_vimp_SubjectName = True
        self.architectureType = 'Res_Unet2'
        self.FirstLayer_FeatureMap_Num = 20

class InputPadding:
    def __init__(self):
        self.Automatic = True
        self.HardDimensions = [116,144,84]

# if Experiments.Index == '8': 
#     InputPadding.HardDimensions = [228,288,168]

code_address = experiment().code_address
class Templatecs:
    def __init__(self):
        self.Image   = code_address + 'general/RigidRegistration' + '/origtemplate.nii.gz'
        self.Mask    = code_address + 'general/RigidRegistration' + '/CropMaskV3.nii.gz'
        self.Address = code_address + 'general/RigidRegistration/'

Template = Templatecs()
