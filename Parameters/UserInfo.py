

Model_Method = 'Cascade'
architectureType = 'Res_Unet2'
gpu = "0"
wmn_csfn = 'wmn' # 'wmn' 'csfn' 
subexperiment_name = 'test_experiment'

#! Preprocessing
class preprocesscs:
    def __init__(self):
        self.Mode = False
        self.BiasCorrection = False
        self.Cropping = False
        self.Reslicing = False

preprocess = preprocesscs()
multi_Class_Mode = True
readAugments_Mode = False

testOnly = False

class experiment:
    def __init__(self):
        self.exp_address = ''
        self.train_address = ''
        self.test_address = ''
        self.init_address = ''


class simulationcs:
    def __init__(self):
        self.TestOnly      = testOnly
        self.epochs        = 5
        self.GPU_Index     = gpu
        self.batch_size    = 10
        self.Use_TestCases_For_Validation = True
        self.ImClosePrediction =  True # False #
        self.Multi_Class_Mode = multi_Class_Mode
        self.LR_Scheduler = True
        self.ReadAugments_Mode = True
    
simulation = simulationcs()

class dataGeneratorcs:
    def __init__(self):
        self.Mode = False
        self.NumSubjects_Per_batch = 5

dataGenerator = dataGeneratorcs()


class InputPaddingcs:
    def __init__(self):
        self.Automatic = True
        self.HardDimensions = [116,144,84]

InputPadding = InputPaddingcs()

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
