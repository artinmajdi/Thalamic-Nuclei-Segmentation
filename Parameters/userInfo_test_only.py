class experiment:

    # Path to the testing data
    test_address = ''

    # Path to the code
    _code_address = ''

    # modality of the input data. wmn / csfn
    image_modality = 'wmn'



class TestOnly:
    # If TRUE , it will run the trained model on test cases.
    mode = True

    """ Address to the main folder holding the trained model.
        This address only applies if mode==True. otherwise it will use the address specified by experiment & subexperiment 
        This directory should point to the parent folder holding on trained models: 
            ACTUAL_TRAINED_MODEL_ADDRESS = model_adress + '/' + FeatureMapNum (e.g. FM20) + '/' + Nucleus_name (e.g. 2-AV) + '/' + Orientation Index (e.g. sd2)
    """
    model_address = ''



# This class specifies while thalamic sides will be analysed
class thalamic_side:
    # Running the network on left thalamus
    left = True

    # Running the network on right thalamus
    right = True

    # This can be left empty. It is used during the training & testing process
    _active_side = ''



class preprocess:
    """ Pre-processing flags
      - Mode             (boolean):   TRUE/FALSE
      - BiasCorrection   (boolean):   Bias Field Correction
      - Cropping         (boolean):   Cropping the input data using the cropped template
      - Reslicing        (boolean):   Re-slicing the input data into the same resolution
      - save_debug_files (boolean):   TRUE/FALSE
      - Normalize        (normalize): Data normalization
    """
    Mode = True
    BiasCorrection = False
    Cropping = True
    Reslicing = True
    save_debug_files = True


class simulation:
    # If TRUE, it will ignore the train data and run the already trained network on test data
    TestOnly = TestOnly()

    # TODO check to see if there is a gpu available
    
    # The GPU card used for training/testing
    GPU_Index = "6"



_code_address = experiment()._code_address



Template = Templatecs()
