class TypeExperimentFuncs():
    def __init__(self):            
        class SubExperimentC: 
            def __init__(self, Index=11, Tag=''):
                self.Index = Index
                self.Tag   = Tag
        self.SubExperimentC = SubExperimentC

        class ReadTrainC:
            def __init__(self, SRI=0 , ET=0 , Main=1):   
                class readAugments: Mode, Tag = True, ''
                self.SRI  = SRI  > 0.5
                self.ET   = ET   > 0.5
                self.Main = Main > 0.5
                self.ReadAugments = readAugments
        self.ReadTrainC = ReadTrainC

        class Transfer_LearningC:
            def __init__(self, Mode=False , FrozenLayers = [0] , Stage = 0):
                self.Mode         = Mode
                self.FrozenLayers = FrozenLayers
                self.Stage        = Stage
        self.Transfer_LearningC = Transfer_LearningC

    def main(self, TypeExperiment = 1):
        switcher = {
            1:  (self.SubExperimentC(Index=11)  ,   self.ReadTrainC(SRI=0 , ET=0 , Main=1)  ,  self.Transfer_LearningC(Mode=False , FrozenLayers=[0]) ),
            2:  (self.SubExperimentC(Index=11)  ,   self.ReadTrainC(SRI=0 , ET=1 , Main=0)  ,  self.Transfer_LearningC(Mode=True  , FrozenLayers=[0]) ),
            3:  (self.SubExperimentC(Index=8)   ,   self.ReadTrainC(SRI=1 , ET=0 , Main=0)  ,  self.Transfer_LearningC(Mode=False , FrozenLayers=[0]) ),          
            }
        return switcher.get(TypeExperiment , 'wrong Index')

Model_Method =  'Cascade' #'FCN_2D' # HCascade' # 
mode3T_7T = '7T'

# TypeExperiment == 1: #  Main
# TypeExperiment == 2: # Transfer Learn ET
# TypeExperiment == 3: # SRI
SubExperiment , ReadTrain , Transfer_Learning = TypeExperimentFuncs().main(1)

class InitializeB:
    FromThalamus   = True
    FromOlderModel = True
    From_3T        = True

class simulation:
    TestOnly      = False
    epochs        = 100
    GPU_Index     = "0,1,2,3"
    Learning_Rate = 1e-3
    num_Layers    = 3
    NormalizaeMethod = 'MinMax' #  '1Std0Mean' #
    nucleus_Index = [8] # ,2,4]
    slicingDim    = [0] # [0,1,2]
    batch_size    = 100
    InputImage2Dvs3D = 2
    FirstLayer_FeatureMap_Num = 20
    verbose = 2
    Multiply_By_Thalmaus = False

    Weighted_Class_Mode = False
    Initialize = InitializeB()
    save_Best_Epoch_Model = True
    Use_Coronal_Thalamus_InSagittal = True
    Use_TestCases_For_Validation = True
    ImClosePrediction = True



    
class InputPadding:
    Automatic = True
    HardDimensions = [1,1,1] # [116,144,84]

class Experiments:
    Index , Tag = '7' , 'cascadeV1'

if Experiments.Index == '8': InputPadding.HardDimensions = [228,288,168]


mode_saveTrue_LoadFalse = True
DropoutValue = 0.3
havingBackGround_AsExtraDimension = True

gapDilation = 5

class Template:
    Image = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/origtemplate.nii.gz'
    Mask = '/array/ssd/msmajdi/code/general/RigidRegistration' + '/CropMaskV3.nii.gz'  # MyCrop_Template2_Gap20


#! metric function
#          1: 'Dice'
#          2: 'Accuracy'
#          3: 'Dice & Accuracy'
MetricIx = 3
Learning_Rate = 1e-3

# if Local_Flag: 
#     Experiments_Address = '/home/artinl/Documents/research'
# else: 
Experiments_Address = '/array/ssd/msmajdi/experiments/keras'


#! Preprocessing
class preprocess:
    Mode = False
    BiasCorrection = False

class normalize:
    Mode = True
    Method = 'MinMax'

AugmentMode = False
Augment_LinearMode = True
Augment_Linear_Length = 6

class Augment_Rotation:
    Mode = True
    AngleMax = '7_6cnts' # '7' # 7_4cnts

class Augment_Shear:
    Mode = False
    ShearMax = 4

Augment_NonLinearMode = False

SaveReportMethod = 'pickle'

