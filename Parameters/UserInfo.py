

class experiment:
    def __init__(self):
        self.exp_address = '/array/ssd/msmajdi/experiments/keras/exp6/'
        self.train_address = '/array/ssd/msmajdi/experiments/keras/exp6/train/'
        self.test_address = '/array/ssd/msmajdi/experiments/keras/exp6/test/'
        self.init_address = ''
        self.subexperiment_name = 'test_01'
        self.ReadAugments_Mode = False


Model_Method = 'Cascade'
architectureType = 'Res_Unet2'
wmn_csfn = 'wmn' # 'wmn' 'csfn' 

class preprocesscs:
    def __init__(self):
        self.Mode = False
        self.BiasCorrection = False
        self.Cropping = False
        self.Reslicing = False

preprocess = preprocesscs()

class simulation:
    def __init__(self):
        self.TestOnly      = False
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
    
class Initialize:
    Modes   = False
    Address = ''

class dataGeneratorcs:
    def __init__(self):
        self.Mode = False
        self.NumSubjects_Per_batch = 5

dataGenerator = dataGeneratorcs()


class InputPadding:
    def __init__(self):
        self.Automatic = True
        self.HardDimensions = [116,144,84]

# if Experiments.Index == '8': 
#     InputPadding.HardDimensions = [228,288,168]


code_address = '/array/ssd/msmajdi/code/thalamus/keras/'
class Templatecs:
    def __init__(self):
        self.Image   = code_address + 'general/RigidRegistration' + '/origtemplate.nii.gz'
        self.Mask    = code_address + 'general/RigidRegistration' + '/CropMaskV3.nii.gz'
        self.Address = code_address + 'general/RigidRegistration/'

Template = Templatecs()
Experiments_Address = '/array/ssd/msmajdi/experiments/keras'

AugmentMode = False
Augment_LinearMode = True
Augment_NonLinearMode = False
Augment_Linear_Length = 6

class Augment_Rotationcs:
    def __init__(self):
        self.Mode = True
        self.AngleMax = 7

Augment_Rotation = Augment_Rotationcs()

class Augment_Shearcs:
    def __init__(self):
        self.Mode = False
        self.ShearMax = 4

Augment_Shear = Augment_Shearcs()
