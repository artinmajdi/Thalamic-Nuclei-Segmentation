
TestOnly = False
class experiment:
    exp_address   = '/array/ssd/msmajdi/experiments/exp6/'
    train_address = '/array/ssd/msmajdi/experiments/data/train/'
    test_address  = '/array/ssd/msmajdi/experiments/data/test/'
    init_address  = ''
    subexperiment_name = 'test_01'
    ReadAugments_Mode  = False
    code_address  = '/array/ssd/msmajdi/code/'


Model_Method = 'Cascade'

wmn_csfn = 'wmn' # 'wmn' 'csfn' 

class preprocesscs:
    def __init__(self):
        self.Mode = True
        self.BiasCorrection = False
        self.Cropping = True
        self.Reslicing = True

preprocess = preprocesscs()

class simulation:
    def __init__(self):
        self.TestOnly      = TestOnly
        self.epochs        = 5
        self.GPU_Index     = "0"
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
    
# simulation2 = simulation()

class Initialize:
    Modes   = False
    Address = ''


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
