"""          Below parameters are set for testing a pre-trained network        """


class experiment:
    # Path to the training data
    train_address = '' # '/array/hdd/msmajdi/data/preprocessed/Siemens/UCLA/CSFn/step2_separated/train/'

    # Path to the testing data
    test_address = '/array/hdd/msmajdi/experiments/data_experiment/CSFn/train/failed_train' # '/array/hdd/msmajdi/data/preprocessed/Siemens/UCLA/CSFn/step2_separated/test' 

    # modality of the input data. wmn / csfn
    image_modality = 'csfn'

    # Reading augmented data. If TRUE, it'll read the data stored inside the subfolder called 'Augments'
    ReadAugments_Mode = False

    # Path to the code
    _code_address = str()

    # Determining if the path to test cases is pointing to a nifti file or the grandparent directory
    _test_path_is_nifti_file = bool()
    _old_test_address = str()

    """         Below two variables are only required for training a new network.
                        It will hold the whole thalamus estimated bounding box for train cases       """

    # Address to the experiment directory. 
    exp_address = '/array/hdd/msmajdi/experiments/exp6/'

    # Subexperiment name
    subexperiment_name = 'GE_Siemens_CSFn_UCLA' # 'GE_Siemens_test_only2'


class TestOnly:
    # If TRUE , it will run the trained model on test cases.
    _mode = bool()

    """ Address to the main folder holding the trained model.
        This address only applies if _mode==True. otherwise it will use the address specified by experiment & subexperiment
        This directory should point to the parent folder holding on trained models:
            ACTUAL_TRAINED_MODEL_ADDRESS = model_adress + '/' + FeatureMapNum (e.g. FM20) + '/' + Nucleus_name (e.g. 2-AV) + '/' + Orientation Index (e.g. sd2)
    """
    model_address = '' # /array/hdd/msmajdi/experiments/exp6/models/GE_Siemens'



"""          Below parameters are set for training a new network         """


class initialize:
    # If TRUE, network weights will be initialized
    mode = True

    # Path to the initialization network. If left empty, the algorithm will use the default path to pretrained models
    init_address = ''


class thalamic_side:
    """ This class specifies while thalamic sides will be analysed """

    # Running the network on left thalamus
    left = True

    # Running the network on right thalamus
    right = True

    # Specifing which thalamic side is the code running on currently
    _active_side = str()


class normalize:
    """ Network initialization

        Mode (boolean): If TRUE, input data will be normalized using the specified method

        Method (str):
          - MinMax: Data will be normalized into 0 minimum & 1 maximum
          - 1Std0Mean: Data will be normalized into 0 minimum & 1 standard deviation
          - Both: Data will be normalized using both above methods
    """

    Mode = True
    Method = '1Std0Mean'


class preprocess:
    """ Pre-processing flags
      - Mode             (boolean):   TRUE/FALSE
      - BiasCorrection   (boolean):   Bias Field Correction
      - Cropping         (boolean):   Cropping the input data using the cropped template
      - Reslicing        (boolean):   Re-slicing the input data into the same resolution
      - save_debug_files (boolean):   TRUE/FALSE
      - Normalize        (normalize): Data normalization
    """
    Mode = False
    BiasCorrection = True
    Cropping = True
    Reslicing = True
    save_debug_files = True
    Normalize = normalize()


class simulation:
    # If TRUE, it will ignore the train data and run the already trained network on test data
    TestOnly = TestOnly()

    # Number of epochs used during training
    epochs = 150

    # The GPU card used for training/testing
    GPU_Index = "1"

    # Batch size used during training
    batch_size = 50

    # If TRUE, it will use test cases for validation during training
    Use_TestCases_For_Validation = True

    # If TRUE, it will perform morphological closing onto the predicted segments
    ImClosePrediction = True

    # If TRUE, it will Use a learning rate scheduler
    LR_Scheduler = True

    # Initial Learning rate
    Learning_Rate = 1e-3

    # Number of layers
    num_Layers = 3

    """ Loss function index
            1: binary_crossentropy
            2: categorical_crossentropy
            3: multi class binary_crossentropy
            4: Logarithm of Dice
            5: Logarithm of Dice + binary_crossentropy
            6: Gmean: Square root of (Logarithm of Dice + binary_crossentropy),
            7: Dice (default)
    """
    lossFunction_Index = 7

    # nuclei indeces
    nucleus_Index = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # slicing orientation. should be left as is
    slicingDim = [2, 1, 0]

    # If TRUE, it will use the network input dimentions obtained from training data for testing data
    use_train_padding_size = False

    # If TRUE, it will only load the subject folders that include "case" in their name
    check_case_SubjectName = False

    # Architecture type
    architectureType = 'Res_Unet2'

    # Number of feature maps for the first layer of Resnet
    FirstLayer_FeatureMap_Num = 20


class InputPadding:
    def __init__(self):
        """ Network Input Dimension
            If Automatic is set to TRUE, it will determine the network input dimention from all training & testing data
            Othewise, it will use the values set in the "HardDimensions" variable
        """
        self.Automatic = True
        self.HardDimensions = [116, 144, 84]


_code_address = experiment()._code_address


class Template:
    """ The path to template nifti image and its corresponding cropping mask

    Args:
        Image   (str): path to original template
        Mask    (Str): path to the cropping mask
        Address (Str): path to the main folder
    """
    Image = _code_address + 'general/RigidRegistration' + '/origtemplate.nii.gz'
    Mask = _code_address + 'general/RigidRegistration' + '/CropMaskV3.nii.gz'
    Address = _code_address + 'general/RigidRegistration/'
