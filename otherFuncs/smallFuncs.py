import json
import os
import sys
from glob import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure
from tqdm import tqdm
from uuid import uuid4 as unique_name_generator
import pathlib
from modelFuncs import Metrics as metrics


def NucleiSelection(ind=1):
    def func_NucleusName(ix):
        if ix == 1:
            NucleusName = '1-THALAMUS'
        elif ix == 2:
            NucleusName = '2-AV'
        elif ix == 4:
            NucleusName = '4-VA'
        elif ix == 5:
            NucleusName = '5-VLa'
        elif ix == 6:
            NucleusName = '6-VLP'
        elif ix == 7:
            NucleusName = '7-VPL'
        elif ix == 8:
            NucleusName = '8-Pul'
        elif ix == 9:
            NucleusName = '9-LGN'
        elif ix == 10:
            NucleusName = '10-MGN'
        elif ix == 11:
            NucleusName = '11-CM'
        elif ix == 12:
            NucleusName = '12-MD-Pf'
        elif ix == 13:
            NucleusName = '13-Hb'
        elif ix == 14:
            NucleusName = '14-MTT'
        return NucleusName

    name = func_NucleusName(ind)
    FullIndexes = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    Full_Names = [func_NucleusName(ix) for ix in FullIndexes]

    return name, FullIndexes, Full_Names


class Nuclei_Class:

    def __init__(self, index=1, method='Cascade'):
        def nucleus_name_func(index):
            switcher = {
                1: '1-THALAMUS',
                2: '2-AV',
                4: '4-VA',
                5: '5-VLa',
                6: '6-VLP',
                7: '7-VPL',
                8: '8-Pul',
                9: '9-LGN',
                10: '10-MGN',
                11: '11-CM',
                12: '12-MD-Pf',
                13: '13-Hb',
                14: '14-MTT'}
            return switcher.get(index, 'wrong index')

        self.name = nucleus_name_func(index)
        self.nucleus_name_func = nucleus_name_func
        self.method = method
        self.index = index
        self.parent, self.child = (None, [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) if self.index == 1 else (1, None)
        self.Indexes = tuple([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        self.Names = [self.nucleus_name_func(index) for index in self.Indexes]

    def All_Nuclei(self):
        class All_Nuclei:
            Indexes = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
            Names = [self.nucleus_name_func(index) for index in Indexes]

        return All_Nuclei()


class Experiment_Folder_Search:
    def __init__(self, General_Address='', Experiment_Name='', subExperiment_Name='', mode='results'):

        class All_Experiments:
            def __init__(self, General_Address):
                self.address = General_Address
                self.List = [s for s in os.listdir(General_Address) if 'exp' in s] if General_Address else []

        class Experiment:
            def __init__(self, Experiment_Name, AllExp_address):
                self.name = Experiment_Name
                self.address = AllExp_address + '/' + Experiment_Name
                self.List_subExperiments = ''
                self.TagsList = []

        def search_AllsubExp_inExp(Exp_address, mode):

            class subExp:
                def __init__(self, name, address, TgC):

                    def find_Planes(address, name, TgC):

                        class SD:
                            def __init__(self, Flag=False, name='', plane_name='', direction_name='', TgC=0,
                                         address=''):
                                self.Flag = Flag
                                if self.Flag:
                                    self.tagList = np.append(['Tag' + str(TgC) + '_' + plane_name], name.split('_'))
                                    self.tagIndex = str(TgC)
                                    if mode == 'results':
                                        self.subject_List = [a for a in os.listdir(address) if 'case' in a]
                                        self.subject_List.sort()
                                    self.name = plane_name
                                    self.direction = direction_name

                        multiPlanar = []
                        sdLst = tuple(['sd0', 'sd1', 'sd2', '2.5D_MV', 'DT', '2.5D_Sum', '1.5D_Sum'])
                        PlNmLs = tuple(['Sagittal', 'Coronal', 'Axial', 'MV', 'DT', '2.5D_Sum', '1.5D_Sum'])
                        if mode == 'results':
                            sdx = os.listdir(address + '/' + name)
                            for sd, plane_name in zip(sdLst,
                                                      PlNmLs):  # ['sd0' , 'sd1' , 'sd2' , '2.5D_MV' , '2.5D_Sum' , '1.5D_Sum']:
                                multiPlanar.append(SD(True, name, sd, plane_name, TgC,
                                                      address + '/' + name + '/' + sd) if sd in sdx else SD())
                        else:
                            for sd, plane_name in zip(sdLst[:3], PlNmLs[
                                                                 :3]):  # [:3]: # ['sd0' , 'sd1' , 'sd2' , '2.5D_MV' , '2.5D_Sum' , '1.5D_Sum']:
                                multiPlanar.append(SD(True, name, sd, plane_name, TgC,
                                                      address + '/' + name + '/' + sd) if sd in sd else SD())
                        return multiPlanar

                    self.name = name
                    self.multiPlanar = find_Planes(address, name, TgC)

            List_subExps = [a for a in os.listdir(Exp_address + '/' + mode) if ('subExp' in a) or ('sE' in a)]

            List_subExps.sort()
            subExps = [subExp(name, Exp_address + '/' + mode, Ix) for Ix, name in enumerate(List_subExps)]
            TagsList = [np.append(['Tag' + str(Ix)], name.split('_')) for Ix, name in enumerate(List_subExps)]

            return subExps, TagsList

        def func_Nuclei_Names():
            NumColumns = 19
            Nuclei_Names = np.append(['subjects'], list(np.zeros(NumColumns - 1)))
            Nuclei_Names[3] = ''

            for nIx, name in enumerate(Nuclei_Class().Indexes, Nuclei_Class().Names):
                Nuclei_Names[nIx] = name

            return Nuclei_Names

        def search_1subExp(List_subExperiments):
            for a in List_subExperiments:
                if a.name == subExperiment_Name:
                    return a

        self.All_Experiments = All_Experiments(General_Address)
        self.Experiment = Experiment(Experiment_Name, self.All_Experiments.address)
        self.Nuclei_Names = func_Nuclei_Names()

        if Experiment_Name:
            self.Experiment.List_subExperiments, self.Experiment.TagsList = search_AllsubExp_inExp(self.Experiment.address, mode)

        if subExperiment_Name:
            self.subExperiment = search_1subExp(self.Experiment.List_subExperiments)


def gpuSetting(GPU_Index: str):
    """ Setting up the selected GPU

        Args:
            GPU_Index (str): GPU card index

        Returns:
            K: session
    """
    assert isinstance(GPU_Index, str), 'GPU index should be a string'

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_Index
    # import tensorflow.compat.v1 as tf1
    # from tensorflow.compat.v1.keras import backend as K
    # # tf.compat.v1.disable_v2_behavior()
    # K.set_session(tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True)))
    return [] # K


def listSubFolders(Dir, params):
    subFolders = [s for s in next(os.walk(Dir))[1] if 'ERROR' not in s]
    if params.WhichExperiment.Dataset.check_case_SubjectName:  subFolders = [s for s in subFolders if 'case' in s]

    subFolders.sort()
    return subFolders


def mkDir(Dir):
    if not os.path.isdir(Dir):
        os.makedirs(Dir)
    return Dir


def saveImage(Image, Affine, Header, outDirectory):
    """ Inputs:  Image , Affine , Header , outDirectory """
    mkDir(outDirectory.split(os.path.basename(outDirectory))[0])
    out = nib.Nifti1Image(Image.astype('float32'), Affine)
    out.get_header = Header
    nib.save(out, outDirectory)


def nibShow(*args):
    if len(args) > 1:
        for ax, im in enumerate(args):
            if ax == 0:
                a = nib.viewers.OrthoSlicer3D(im, title=str(ax))
            else:
                b = nib.viewers.OrthoSlicer3D(im, title='2')
                a.link_to(b)
    else:
        a = nib.viewers.OrthoSlicer3D(args[0])
    a.show()


def fixMaskMinMax(Image, name):
    if Image.max() > 1 or Image.min() < 0:
        print('Error in label values', 'min', Image.min(), 'max', Image.max(), '    ', name.split('_PProcessed')[0])
        Image = np.float32(Image)
        Image = (Image - Image.min()) / (Image.max() - Image.min())

    return Image


def terminalEntries(UserInfo):

    def single_nifti_test_case_directory_correction(experiment_class):
        """ This function checks the test directory. If it points to an individual nifti file;
            it creates a folder with a unique name and move the nifti file into that folder """
    
        experiment_class.test_path_is_nifti_file = False
        if '.nii' in experiment_class.test_address:
            experiment_class.test_path_is_nifti_file = True
            old_test_file_address = experiment_class.test_address
            file_main_directory   = os.path.dirname(old_test_file_address)
            file_name             = os.path.basename(old_test_file_address)
            unique_folder_name    = str(unique_name_generator())
            new_test_address      = mkDir( file_main_directory + '/' + unique_folder_name +'/case_1')
            shutil.move( old_test_file_address, new_test_address + '/' + file_name )

            experiment_class.test_address     = file_main_directory + '/' + unique_folder_name
            experiment_class.old_test_address = file_main_directory

        return experiment_class

    for en in range(len(sys.argv)):
        entry = sys.argv[en]

        if entry.lower() in ('-g', '--gpu'):  # gpu num
            UserInfo['simulation'].GPU_Index = sys.argv[en + 1]

        elif entry in ('--train'):
            UserInfo['experiment'].train_address = os.path.abspath(sys.argv[en + 1]) 

        elif entry in ('--test'):
            UserInfo['experiment'].test_address =  os.path.abspath(sys.argv[en + 1]) 

        elif entry in ('--model'):
            UserInfo['simulation'].TestOnly.model_address = os.path.abspath(sys.argv[en + 1]) 

        elif entry in ('--modality'):
            UserInfo['experiment'].image_modality = sys.argv[en + 1].lower()


    # Checks the path to test files to see if it points to a single nifti file or a parent folder consist of multiple test cases
    UserInfo['experiment'] = single_nifti_test_case_directory_correction(UserInfo['experiment'])

    # setting test-only to TRUE if no address to traininig data was provided
    if not UserInfo['experiment'].train_address:
        UserInfo['simulation'].TestOnly.mode = True

    # Settig up the address to the main code
    UserInfo['experiment'].code_address = str(pathlib.Path(__file__).parent.parent)


    # Setting the GPU
    if UserInfo['simulation'].GPU_Index:
        os.environ["CUDA_VISIBLE_DEVICES"] = UserInfo['simulation'].GPU_Index


    UserInfo['simulation'] = UserInfo['simulation']()

    return UserInfo


def search_ExperimentDirectory(whichExperiment):
    SD = '/sd' + str(whichExperiment.Dataset.slicingInfo.slicingDim)
    FM    = '/FM' + str(whichExperiment.HardParams.Model.Layer_Params.FirstLayer_FeatureMap_Num)
    NN    = '/'   + whichExperiment.Nucleus.name
    code_address   = dir_check(whichExperiment.Experiment.code_address)
    Exp_address    = dir_check(whichExperiment.Experiment.exp_address)
    subexperiment_name = whichExperiment.Experiment.subexperiment_name
    NucleusName    = whichExperiment.Nucleus.name

    def checkInputDirectory(Dir, NucleusName, sag_In_Cor, modeData):
        """ Check input directory for each subject

        Args:
            Dir (str):            Input directory
            NucleusName (str):    Nucleus name
            sag_In_Cor (boolean): If TRUE, it will copy the predicted whole thalamus in the coronal network, inside the sagital network results
            modeData (boolean):   Specified if this is ran on test or train dataset 
        """        

        def Search_ImageFolder(Dir, NucleusName):

            def splitNii(s):
                return s.split('.nii.gz')[0]

            def Classes_Local(Dir):
                class deformation:
                    address = ''
                    testWarp = ''
                    testInverseWarp = ''
                    testAffine = ''

                class temp:
                    CropMask = ''
                    Cropped = ''
                    BiasCorrected = ''
                    Deformation = deformation
                    address = ''

                class tempLabel:
                    address = ''
                    Cropped = ''

                class label:
                    LabelProcessed = ''
                    LabelOriginal = ''
                    Temp = tempLabel
                    address = ''

                class newCropInfo:
                    OriginalBoundingBox = ''
                    PadSizeBackToOrig = ''

                class Files:
                    ImageOriginal = ''
                    ImageProcessed = ''
                    Label = label
                    Temp = temp
                    address = Dir
                    NewCropInfo = newCropInfo
                    subjectName = ''

                return Files

            def search_inside_subject_folder_label(Files, NucleusName):
                """ Searching inside the Label folder

                Args:
                    Files      (File): List of all files inside the subject folder
                    NucleusName (str): Name of the searched nucleus

                Returns:
                    Files      (File): Updated list of files inside the subject folder
                """        

                A = next(os.walk(Files.Label.address))

                # Single Class Mode. This will run on the first network of cascade algorithm (whole thalamus)
                if 'MultiClass' not in NucleusName:

                    # Checking to see if the nucleus nifti file and its pre-processed nifti file exist
                    for s in A[2]:

                        if NucleusName + '_PProcessed.nii.gz' in s:
                            Files.Label.LabelProcessed = s

                        elif NucleusName + '.nii.gz' in s:
                            Files.Label.LabelOriginal = s

                    # If the nifti pre-processed file for the searched nucleus didn't exist, this will duplicate 
                    # the original nucleus nifti file and name the second indstance as nucleu name plus PProcessed
                    if not Files.Label.LabelProcessed:

                        Files.Label.LabelProcessed = NucleusName + '_PProcessed.nii.gz'

                        shutil.copyfile(Files.Label.address + '/' + NucleusName + '.nii.gz',
                                 Files.Label.address + '/' + NucleusName + '_PProcessed.nii.gz')

                    # Checking the temp folder inside the Label-temp folder
                    cpd_lst = glob(Files.Label.Temp.address + '/' + NucleusName + '_Cropped.nii.gz')

                    if cpd_lst:  Files.Label.Temp.Cropped = NucleusName + '_Cropped'

                # Multi Class mode. This will run on the second network of cascade algorithm
                else:
                    # Looping through all nuclei
                    for nucleus in whichExperiment.Nucleus.FullNames:

                        # If the original nucleus nifti file exists, but the pre-processed one doesn't, this will dupicate 
                        # the original nucleu nifti file and rename the second instance as the name of nucleus plus PProcessed
                        if (nucleus + '.nii.gz' in A[2]) and (nucleus + '_PProcessed.nii.gz' not in A[2]):

                            shutil.copyfile(Files.Label.address + '/' + nucleus + '.nii.gz',
                                     Files.Label.address + '/' + nucleus + '_PProcessed.nii.gz')

                return Files

            def search_inside_subject_folder_temp(Files):
                A = next(os.walk(Files.Temp.address))
                for s in A[2]:
                    if 'CropMask.nii.gz' in s:
                        Files.Temp.CropMask = splitNii(s)
                    elif 'bias_corr.nii.gz' in s:
                        Files.Temp.BiasCorrected = splitNii(s)
                    elif 'bias_corr_Cropped.nii.gz' in s:
                        Files.Temp.Cropped = splitNii(s)
                    else:
                        Files.Temp.origImage = splitNii(s)

                if 'deformation' in A[1]:
                    Files.Temp.Deformation.address = Files.Temp.address + '/deformation'
                    B = next(os.walk(Files.Temp.Deformation.address))
                    for s in B[2]:
                        if 'testWarp.nii.gz' in s:
                            Files.Temp.Deformation.testWarp = splitNii(s)
                        elif 'testInverseWarp.nii.gz' in s:
                            Files.Temp.Deformation.testInverseWarp = splitNii(s)
                        elif 'testAffine.txt' in s:
                            Files.Temp.Deformation.testAffine = splitNii(s)

                if not Files.Temp.Deformation.address:
                    Files.Temp.Deformation.address = mkDir(Files.Temp.address + '/deformation')

                return Files

            def search_inside_subject_folder(Files):

                # List of nifti files inside the searched folder
                files_folders_list = os.listdir(Files.address)
                nifti_files_list = [n for n in files_folders_list if '.nii.gz' in n]

                # Searching through all nifri images found inside the subject folder
                for s in nifti_files_list:

                    # Setting the ImageProcessed flag if it finds a nifti image inside the designated folder with PProcessed as part of its name
                    if 'PProcessed.nii.gz' in s:
                        Files.ImageProcessed     = s.split('.nii.gz')[0]

                    # Setting the addresses to ImageOriginal, temp subfolder address, Label subaddress and Label-temp subfolder
                    else:
                        Files.ImageOriginal      = s.split('.nii.gz')[0]
                        Files.Temp.address       = mkDir(Files.address + '/temp')


                # If the Label subfolder exist, this will create a folder named temp inside the Label subfolder
                if 'Label' in files_folders_list:
                    Files.Label.address      = Files.address + '/Label'
                    Files.Label.Temp.address = mkDir(Files.address + '/Label/temp')


                # If a nifti image exist inside the searched folder, but a PProcessed file doesn't, this will duplicate the original image and name it as PProcessed.nii.gz 
                if Files.ImageOriginal and (not Files.ImageProcessed):
                    Files.ImageProcessed = 'PProcessed'
                    shutil.copyfile(Dir + '/' + Files.ImageOriginal + '.nii.gz', Dir + '/PProcessed.nii.gz')

                return Files

            Files = Classes_Local(Dir)
            Files = search_inside_subject_folder(Files)

            # Checking to make sure an nifti image exist inside the folder
            if Files.ImageOriginal:
                # Searching inside the Label subfolder
                if os.path.exists(Files.Label.address): 
                    Files = search_inside_subject_folder_label(Files, NucleusName)

                # Searching inside the temp subfolder inside the subject directory
                if os.path.exists(Files.Temp.address):  
                    Files = search_inside_subject_folder_temp(Files)

            return Files

        class Input:
            address = os.path.abspath(Dir)
            Subjects = {}

        def LoopReadingData(Input, Dirr):
            if os.path.exists(Dirr):
                SubjectsList = next(os.walk(Dirr))[1]

                if whichExperiment.Dataset.check_case_SubjectName:
                    SubjectsList = [s for s in SubjectsList if ('case' in s)]

                for s in SubjectsList:
                    Input.Subjects[s] = Search_ImageFolder(Dirr + '/' + s, NucleusName)
                    Input.Subjects[s].subjectName = s

            return Input

        # This "if" statement skips the train dataset if the TestOnly flag is set to TRUE
        if not (modeData == 'train' and whichExperiment.TestOnly.mode):

            # Setting the adress to the full dataset
            if modeData == 'train':
                Dir = whichExperiment.Experiment.train_address 
            else:
                Dir = whichExperiment.Experiment.test_address
            
            # Finding all subjects inside the specified dataset
            Input = LoopReadingData(Input, Dir)

            # Loading the augmented data saved inside a subfolder called 'Augments' inside the train folder
            if whichExperiment.Experiment.ReadAugments_Mode and not (modeData == 'test'):
                Input = LoopReadingData(Input, Dir + '/Augments/')

        return Input

    def add_Sagittal_Cases(whichExperiment, train, test, NucleusName):
        if whichExperiment.Nucleus.Index[0] == 1 and whichExperiment.Dataset.slicingInfo.slicingDim == 2:
            train.Input_Sagittal = checkInputDirectory(train.address, NucleusName, True, 'train')
            test.Input_Sagittal = checkInputDirectory(test.address, NucleusName, True, 'test')
        return train, test
       
    # Setting the address to the trained model based on the number of featuremaps, nucleus name, and orientation
    if whichExperiment.TestOnly.mode:

        # if an address to a trained model is provided, that will be used to segment the test cases
        if whichExperiment.TestOnly.model_address:
            model_address = dir_check(whichExperiment.TestOnly.model_address) + FM + NN + SD

        # if an address to a trained model isn't provided, the default trained models will be used to segment the test cases
        else:
            # Setting the image modality: WMn / CSFn
            net_name =  whichExperiment.Experiment.image_modality.lower() 

            # The address to default trained models based on the user entered input image modality
            model_address = code_address + 'Trained_Models/' + net_name + FM + NN + SD

    else:
        # In case of training a network, the newly trained network will be used to segment the test cases
        model_address = Exp_address + 'models/' + subexperiment_name + FM + NN + SD

    class train:
        address = Exp_address + 'train'
        Model = model_address
        model_Tag = ''
        Input = checkInputDirectory(address, NucleusName, False, 'train')

    class test:
        address = Exp_address + 'test'
        Result = Exp_address + 'results/' + subexperiment_name + SD
        Input = checkInputDirectory(address, NucleusName, False, 'test')

    train, test = add_Sagittal_Cases(whichExperiment, train, test, NucleusName)

    class Directories:
        Train = train()
        Test = test()

    return Directories()


def imShow(*args):
    _, axes = plt.subplots(1, len(args))
    for ax, im in enumerate(args):
        axes[ax].imshow(im, cmap='gray')

    # a = nib.viewers.OrthoSlicer3D(im,title='image')

    plt.show()

    return True


def mDice(msk1, msk2):
    intersection = msk1 * msk2
    return intersection.sum() * 2 / (msk1.sum() + msk2.sum() + np.finfo(float).eps)


def findBoundingBox(PreStageMask):
    if not PreStageMask.max():
        PreStageMask = np.ones(PreStageMask.shape)

    objects = measure.regionprops(measure.label(PreStageMask))

    L = len(PreStageMask.shape)
    if len(objects) > 1:
        area = []
        for obj in objects: area = np.append(area, obj.area)

        Ix = np.argsort(area)
        bbox = objects[Ix[-1]].bbox

    else:
        bbox = objects[0].bbox

    BB = [[bbox[d], bbox[L + d]] for d in range(L)]

    return BB


def Saving_UserInfo(DirSave, params):
    User_Info = {
        'num_Layers': int(params.WhichExperiment.HardParams.Model.num_Layers),
        'Learning_Rate': params.UserInfo['simulation'].Learning_Rate,
        'slicing_Dim': params.UserInfo['simulation'].slicingDim[0],
        'batch': int(params.UserInfo['simulation'].batch_size),
        'InputPadding_Mode': params.UserInfo['InputPadding']().Automatic,
        'InputPadding_Dims': [int(s) for s in params.WhichExperiment.HardParams.Model.InputDimensions],
    }
    mkDir(DirSave)
    with open(DirSave + '/UserInfo.json', "w") as j:
        j.write(json.dumps(User_Info))


def closeMask(mask, cnt):
    struc = ndimage.generate_binary_structure(3, 2)
    if cnt > 1: struc = ndimage.iterate_structure(struc, cnt)
    return ndimage.binary_closing(mask, structure=struc)


def dir_check(directory):
    return os.path.abspath(directory) + '/'


def apply_MajorityVoting(params):
    def func_manual_label(subject, nucleusNm):
        class manualCs:
            def __init__(self, flag, label):
                self.Flag = flag
                self.Label = label

        dirr = subject.address + '/Label/' + nucleusNm + '_PProcessed.nii.gz'

        if os.path.isfile(dirr):
            label = fixMaskMinMax(nib.load(dirr).get_fdata(), nucleusNm)
            manual = manualCs(flag=True, label=label)

        else:
            manual = manualCs(flag=False, label='')

        return manual

    # address = params.WhichExperiment.Experiment.exp_address + '/results/' + params.WhichExperiment.Experiment.subexperiment_name + '/'

    a = Nuclei_Class().All_Nuclei()
    num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes

    for sj in tqdm(params.directories.Test.Input.Subjects):
        subject = params.directories.Test.Input.Subjects[sj]
        address = subject.address + '/' + params.UserInfo['thalamic_side'].active_side + '/'

        VSI, Dice, HD = np.zeros((num_classes, 2)), np.zeros((num_classes, 2)), np.zeros((num_classes, 2))
        for cnt, (nucleusNm, nucleiIx) in enumerate(zip(a.Names, a.Indexes)):

            ix, pred3Dims = 0, ''
            im = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')

            manual = func_manual_label(subject, nucleusNm)
            for sdInfo in ['sd0', 'sd1', 'sd2']:
                address_nucleus = address + sdInfo + '/' + nucleusNm + '.nii.gz'
                if os.path.isfile(address_nucleus):
                    pred = nib.load(address_nucleus).get_fdata()[..., np.newaxis]
                    pred3Dims = pred if ix == 0 else np.concatenate((pred3Dims, pred), axis=3)
                    ix += 1

            if ix > 0:
                predMV = pred3Dims.sum(axis=3) >= 2
                saveImage(predMV, im.affine, im.header, address + '2.5D_MV/' + nucleusNm + '.nii.gz')

                if manual.Flag:
                    VSI[cnt, :] = [nucleiIx, metrics.VSI_AllClasses(predMV, manual.Label).VSI()]
                    HD[cnt, :] = [nucleiIx, metrics.HD_AllClasses(predMV, manual.Label).HD()]
                    Dice[cnt, :] = [nucleiIx, mDice(predMV, manual.Label)]

        if Dice[:, 1].sum() > 0:
            np.savetxt(address + '2.5D_MV/VSI_All.txt', VSI, fmt='%1.1f %1.4f')
            np.savetxt(address + '2.5D_MV/HD_All.txt', HD, fmt='%1.1f %1.4f')
            np.savetxt(address + '2.5D_MV/Dice_All.txt', Dice, fmt='%1.1f %1.4f')


def extracting_the_biggest_object(pred_Binary):
    objects = measure.regionprops(measure.label(pred_Binary))

    # L = len(pred_Binary.shape)
    if len(objects) > 1:
        area = []
        for obj in objects: area = np.append(area, obj.area)

        Ix = np.argsort(area)
        obj = objects[Ix[-1]]

        fitlered_pred = np.zeros(pred_Binary.shape)
        for cds in obj.coords:
            fitlered_pred[tuple(cds)] = True

        return fitlered_pred

    else:
        return pred_Binary


def test_precision_recall():
    import pandas as pd

    directory = '/array/ssd/msmajdi/experiments/keras/exp6/results/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/sd2/vimp2_967_08132013_KW/'
    directoryM = '/array/ssd/msmajdi/experiments/keras/exp6/crossVal/Main/a/vimp2_967_08132013_KW/Label/'

    Names = Nuclei_Class(index=1, method='Cascade').All_Nuclei().Names

    write_flag = False
    PR = {}
    if write_flag:
        writer = pd.ExcelWriter(path=directory + 'Precision_Recall.xlsx', engine='xlsxwriter')

    for ind in range(13):

        nucleus_name = Names[ind].split('-')[1]
        msk = nib.load(directory + Names[ind] + '.nii.gz').get_fdata()
        mskM = nib.load(directoryM + Names[ind] + '_PProcessed.nii.gz').get_fdata()

        # plt.plot(np.unique(msk))

        precision, recall = metrics.Precision_Recall_Curve(y_true=mskM, y_pred=msk, Show=True, name=nucleus_name,
                                                           directory=directory)

        if write_flag:
            df = pd.DataFrame.from_dict({'precision': precision, 'recall': recall})
            df.to_excel(writer, sheet_name=nucleus_name)

    if write_flag: writer.save()


def test_extract_biggest_object():
    from skimage import measure
    import os
    from tqdm import tqdm

    dir_predictions = '/array/ssd/msmajdi/experiments/keras/exp6_uncropped/results/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CVs_all/sd2/'
    main_directory = '/array/ssd/msmajdi/experiments/keras/exp6_uncropped/crossVal/Main/'  # c/vimp2_988_08302013_CB/PProcessed.nii.gz'

    for cv in tqdm(['a/', 'b/', 'c/', 'd/', 'e/', 'f/']):

        if not os.path.exists(main_directory + cv): continue

        subjects = [s for s in os.listdir(main_directory + cv) if 'case' in s]
        for subj in tqdm(subjects):

            # im = nib.load(main_directory + cv + subj + '/PProcessed.nii.gz').get_fdata()
            label = nib.load(main_directory + cv + subj + '/Label/1-THALAMUS_PProcessed.nii.gz').get_fdata()
            label = fixMaskMinMax(label, subj)
            OP = nib.load(dir_predictions + subj + '/1-THALAMUS.nii.gz')
            original_prediction = OP.get_fdata()

            objects = measure.regionprops(measure.label(original_prediction))

            if len(objects) > 1:
                area = []
                for obj in objects:
                    area = np.append(area, obj.area)

                Ix = np.argsort(area)
                obj = objects[Ix[-1]]

                filtered_prediction = np.zeros(original_prediction.shape)
                for cds in obj.coords:
                    filtered_prediction[tuple(cds)] = True

                Dice = np.zeros(2)
                Dice[0], Dice[1] = 1, mDice(filtered_prediction > 0.5, label > 0.5)
                np.savetxt(dir_predictions + subj + '/Dice_1-THALAMUS_biggest_obj.txt', Dice, fmt='%1.4f')

                saveImage(filtered_prediction, OP.affine, OP.header,
                                     dir_predictions + subj + '/1-THALAMUS_biggest_obj.nii.gz')


class SNR_experiment:

    def __init__(self):
        pass

    def script_adding_WGN(self, directory='/mnt/sda5/RESEARCH/PhD/Thalmaus_Dataset/SNR_Tests/vimp2_ANON695_03132013/',
                          SD_list=np.arange(1, 80, 5), run_network=True):

        def add_WGN(input_image=[], noise_mean=0, noise_std=1):
            from numpy.fft import fftshift, fft2
            gaussian_noise_real = np.random.normal(loc=noise_mean, scale=noise_std, size=input_image.shape)
            gaussian_noise_imag = np.random.normal(loc=noise_mean, scale=noise_std, size=input_image.shape)
            gaussian_noise = gaussian_noise_real + 1j * gaussian_noise_imag

            # template = nib.load('general/RigidRegistration/cropped_origtemplate.nii.gz').get_fdata()
            # psd_template = np.mean(abs(fftshift(fft2(template/template.max())))**2)

            # psd_template, max_template = 325, 12.66
            psd_signal = np.mean(abs(fftshift(fft2(input_image))) ** 2)

            # psd_noise_image = psd_signal - psd_template * (input_image.max()**2)
            psd_noise = np.mean(abs(fftshift(fft2(gaussian_noise))) ** 2)
            # psd_noise       = psd_noise_added + psd_noise_image

            SNR = 10 * np.log10(psd_signal / psd_noise)  # SNR = PSD[s]/PSD[n]  if x = s + n
            # SNR = 10*np.log10(np.mean(input_image**2)/np.mean(abs(gaussian_noise)**2))  # SNR = E[s^2]/E[n^2]  if x = s + n
            return abs(input_image + gaussian_noise), SNR

        directory2 = os.path.dirname(directory).replace(' ', '\ ')
        os.system("mkdir {0}/vimp2_orig_SNR_10000 ; mv {0}/* {0}/vimp2_orig_SNR_10000/ ".format(directory2))
        subject_name = 'vimp2_orig'
        dir_original_image = directory + subject_name + '_SNR_10000'

        SNR_List = []
        np.random.seed(0)
        for noise_std in SD_list:

            imF = nib.load(dir_original_image + '/PProcessed.nii.gz')
            im = imF.get_fdata()

            noisy_image, SNR = add_WGN(input_image=im, noise_mean=0, noise_std=noise_std)
            SNR = int(round(SNR))

            if SNR not in SNR_List:
                SNR_List.append(SNR)
                print('SNR:', int(round(SNR)), 'std:', noise_std)

                dir_noisy_image = directory + 'vimp2_noisy_SNR_' + str(SNR)
                saveImage(noisy_image, imF.affine, imF.header, dir_noisy_image + '/PProcessed.nii.gz')

                os.system('cp -r %s %s' % (
                dir_original_image.replace(' ', '\ ') + '/Label', dir_noisy_image.replace(' ', '\ ') + '/Label'))

                if run_network:
                    os.system("python main.py --test %s" % (dir_noisy_image.replace(' ', '\ ') + '/PProcessed.nii.gz'))

    def read_all_Dices_and_SNR(self, directory=''):

        Dices = {}
        for subj in [s for s in os.listdir(directory) if os.path.isdir(directory + s)]:
            SNR = int(subj.split('_SNR_')[-1])

            Dices[SNR] = pd.read_csv(directory + subj + '/left/2.5D_MV/Dice_All.txt', index_col=0, header=None,
                                     delimiter=' ', names=[SNR]).values.reshape([-1]) if os.path.isfile(
                directory + subj + '/left/2.5D_MV/Dice_All.txt') else np.zeros(13)

        df = pd.DataFrame(Dices, index=Thalamus_Sub_Functions().All_Nuclei().Names)
        df = df.transpose()
        df.columns.name = 'nucleus'
        df.index.name = 'SNR'
        df = df.sort_values(['SNR'], ascending=[False])
        df.to_csv(directory + 'Dice_vs_SNR.csv')

        return df

    def loop_all_subjects_read_Dice_SNR(self, directory='/mnt/sda5/RESEARCH/PhD/Thalmaus_Dataset/SNR_Tests/'):

        writer = pd.ExcelWriter(directory + 'Dice_vs_SNR.xlsx', engine='xlsxwriter')
        for subject in [s for s in os.listdir(directory) if os.path.isdir(directory + s) and 'case_' in s]:
            print(subject)
            df = self.read_all_Dices_and_SNR(directory=directory + subject + '/')
            df.to_excel(writer, sheet_name=subject)

        writer.save()


class Thalamus_Sub_Functions:
    def __init__(self):
        pass

    def measure_metrics(self, Dir_manual='', Dir_prediction='', metrics=['DICE'], save=False):

        if not (os.path.isdir(Dir_prediction) and os.path.isdir(Dir_manual)): raise Warning(
            'directory does not exist'.upper())
        Measurements = {s: [] for s in metrics}
        for nuclei in Nuclei_Class().All_Nuclei().Names:
            pred = nib.load(Dir_prediction + nuclei + '.nii.gz').get_fdata()
            manual = nib.load(Dir_manual + nuclei + '_PProcessed.nii.gz').get_fdata()

            if 'DICE' in metrics: Measurements['DICE'].append([nuclei, mDice(pred, manual)])

        if save:
            for mt in metrics: np.savetxt(Dir_prediction + 'All_' + mt + '.txt', Measurements[mt], fmt='%1.1f %1.4f')

        return Measurements

    def nucleus_name(self, index=1):
        switcher = {
            1: '1-THALAMUS',
            2: '2-AV',
            4: '4-VA',
            5: '5-VLa',
            6: '6-VLP',
            7: '7-VPL',
            8: '8-Pul',
            9: '9-LGN',
            10: '10-MGN',
            11: '11-CM',
            12: '12-MD-Pf',
            13: '13-Hb',
            14: '14-MTT'}
        return switcher.get(index, 'wrong index')

    def All_Nuclei(self):
        indexes = tuple([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        class All_Nuclei:
            Indexes = indexes[:]
            Names = [self.nucleus_name(index) for index in Indexes]

        return All_Nuclei()

    def run_network(self, directory='mnt/PProcessed.nii.gz', thalamic_side='--left', modality='--wmn', gpu="None"):
        os.system('python main.py --test %s %s %s --gpu %s' % (directory, thalamic_side, modality, gpu))


def orientation_name_correction(orientation):
    names = {
        'sd0': 'Sagittal',
        'sd1': 'Coronal',
        'sd2': 'Axial',
        '2.5D_MV': 'Majority Voting',
    }

    return names[orientation]