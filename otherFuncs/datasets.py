import os
import shutil
import sys
from random import shuffle
import nibabel as nib
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import otherFuncs.smallFuncs as smallFuncs
import preprocess.normalizeA as normalizeA


def ClassesFunc():
    class ImageLabel:
        Image = np.zeros(3)
        Mask = ''

    class info:
        Height = ''
        Width = ''

    class data:
        Train = ImageLabel()
        Train_ForTest = ""
        Test = ""
        Validation = ImageLabel()
        Info = info()
        Sagittal_Train_ForTest = ""
        Sagittal_Test = ""

    class trainCase:
        def __init__(self, Image, Mask):
            self.Image = Image
            self.Mask = Mask

    class testCase:
        def __init__(self, Image, Mask, OrigMask, Affine, Header, original_Shape):
            self.Image = Image
            self.Mask = Mask
            self.OrigMask = OrigMask
            self.Affine = Affine
            self.Header = Header
            self.original_Shape = original_Shape

    return ImageLabel, data, trainCase, testCase


ImageLabel, data, trainCase, testCase = ClassesFunc()

"""
def DatasetsInfo(DatasetIx):
    switcher = {
        1: ('SRI_3T', '/array/ssd/msmajdi/data/preProcessed/3T/SRI_3T'),
        2: ('SRI_ReSliced', '/array/ssd/msmajdi/data/preProcessed/3T/SRI_ReSliced'),
        3: ('croppingData', '/array/ssd/msmajdi/data/preProcessed/croppingData'),
        4: ('All_7T', '/array/ssd/msmajdi/data/preProcessed/7T/All_DBD'),
        5: ('20priors', '/array/ssd/msmajdi/data/preProcessed/7T/20priors'),
    }
    return switcher.get(DatasetIx, 'WARNING: Invalid dataset index')
"""

def paddingNegativeFix(sz, Padding):
    padding = np.array([list(x) for x in Padding])
    crd = -1 * padding
    padding[padding < 0] = 0
    crd[crd < 0] = 0
    Padding = tuple([tuple(x) for x in padding])

    for ix in range(3): 
        crd[ix, 1] = sz[ix] if crd[ix, 1] == 0 else -crd[ix, 1]

    return padding, crd


def loadDataset(params):
    """    Loading the dataset    """    
    def inputPreparationForUnet(im, subject2, params):

        def CroppingInput(im, Padding2):
            if np.min(Padding2) < 0:
                Padding2, crd = paddingNegativeFix(im.shape, Padding2)
                im = im[crd[0, 0]:crd[0, 1], crd[1, 0]:crd[1, 1], crd[2, 0]:crd[2, 1]]

            return np.pad(im, Padding2[:3], 'constant')

        im = CroppingInput(im, subject2.Padding)
        im = np.transpose(im, [2, 0, 1])
        im = np.expand_dims(im, axis=3).astype('float32')

        return im

    def read_cropped_inputs(params, subject, DIRECTORY):
        inputF = nib.load(DIRECTORY)
        if 1 not in params.WhichExperiment.Nucleus.Index:

            BB = subject.NewCropInfo.OriginalBoundingBox[
                params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse]
            input_im = inputF.dataobj[BB[0][0]:BB[0][1], BB[1][0]:BB[1][1], BB[2][0]:BB[2][1]]
        else:
            input_im = inputF.get_data()

        return inputF, input_im

    def readingImage(params, subject2):

        def readingWithTranpose(DIRECTORY):
            ImageF, Image = read_cropped_inputs(params, subject2, DIRECTORY)
            return ImageF, np.transpose(Image, params.WhichExperiment.Dataset.slicingInfo.slicingOrder)

        imF, im = readingWithTranpose(subject2.address + '/' + subject2.ImageProcessed + '.nii.gz')
        im = inputPreparationForUnet(im, subject2, params)
        im = normalizeA.main_normalize(params.preprocess.Normalize, im)

        return im, imF

    def readingNuclei(params, subject, imFshape):

        def backgroundDetector(masks):
            a = np.sum(masks, axis=3)
            background = np.zeros(masks.shape[:3])
            background[np.where(a == 0)] = 1
            background = np.expand_dims(background, axis=3)
            return background

        def readingOriginalMask(NucInd):
            nameNuclei, _, _ = smallFuncs.NucleiSelection(NucInd)
            inputMsk = subject.Label.address + '/' + nameNuclei + '_PProcessed.nii.gz'

            if os.path.exists(inputMsk):
                _, mask = read_cropped_inputs(params, subject, inputMsk)
                mask = smallFuncs.fixMaskMinMax(mask, nameNuclei)
            else:
                mask = np.zeros(10, 10, 10)

            return np.expand_dims(mask, axis=3)

        for cnt, NucInd in enumerate(params.WhichExperiment.Nucleus.Index):
            origMsk1N = readingOriginalMask(NucInd)

            msk1N = np.transpose(np.squeeze(origMsk1N), params.WhichExperiment.Dataset.slicingInfo.slicingOrder)
            msk1N = inputPreparationForUnet(msk1N, subject, params)

            origMsk = origMsk1N if cnt == 0 else np.concatenate((origMsk, origMsk1N), axis=3).astype('float32')
            msk = msk1N if cnt == 0 else np.concatenate((msk, msk1N), axis=3).astype('float32')

        if params.WhichExperiment.HardParams.Model.Method.havingBackGround_AsExtraDimension:
            background = backgroundDetector(msk)
            msk = np.concatenate((msk, background), axis=3).astype('float32')

        return origMsk, msk

    def Error_MisMatch_In_Dim_ImageMask(subject, mode, nameSubject):
        AA = subject.address.split(os.path.dirname(subject.address) + '/')
        shutil.move(subject.address, os.path.dirname(subject.address) + '/' + 'ERROR_' + AA[1])
        print('WARNING:', mode, nameSubject, ' image and mask have different shape sizes')

    def main_ReadingDataset(params):

        def sagittalFlag():
            """
            The flag for whether running the trained coronal network to segment whole thalamus to be 
            used in the 2nd network of sagittal network.

            Returns:
                (boolean): Returns True, if the coronal network is running on whole thalamus.
            """            
            slicingDim = params.WhichExperiment.Dataset.slicingInfo.slicingDim
            nucleus_index = params.WhichExperiment.Nucleus.Index[0]
            return (nucleus_index == 1) and (slicingDim == 2)

        def trainFlag():
            """
            Returns:
                (boolean): True if the network is not set on test only and train subject list is not empty
            """

            Flag_TestOnly = params.WhichExperiment.TestOnly
            Flag_notEmpty = params.directories.Train.Input.Subjects
            measure_train = params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data

            return (not Flag_TestOnly or measure_train) and Flag_notEmpty

        Th = 0.5 * params.WhichExperiment.HardParams.Model.LabelMaxValue

        def separatingConcatenatingIndexes(Data=[], sjList=[], mode='train'):
            """
            Preparing the training & validation dataset by concatenating all 2D slices in all 3D input volumes

            Args:
                Data: Input data
                sjList (list): Subject list
                mode (boolean): Training or validation 

            Returns:
                train_data: Concatenated training data
            """

            # Finding the total number of 2D slices used for training
            Sz0 = 0
            for nameSubject in sjList: 
                Sz0 += Data[nameSubject].Image.shape[0]
            
            # In order to expedite the loading process, initially an empty array is built 
            images = np.zeros((tuple([Sz0]) + Data[list(Data)[0]].Image.shape[1:]))
            masks = np.zeros((tuple([Sz0]) + Data[list(Data)[0]].Mask.shape[1:]))

            # Concatenating the training data into one array
            d1 = 0
            for _, nameSubject in enumerate(tqdm(sjList, desc='concatenating: ' + mode)):
                im, msk = Data[nameSubject].Image, Data[nameSubject].Mask

                images[d1:d1 + im.shape[0], ...] = im
                masks[d1:d1 + im.shape[0], ...] = msk
                d1 += im.shape[0]

            return trainCase(Image=images, Mask=masks.astype('float32'))

        def separateTrainVal_and_concatenateTrain(DataAll):
            """
            Separating the training & validation data

            Args:
                DataAll : loaded input dataset

            Returns:
                DataAll: Updated input dataset, with validation & training separated 
            """            

            TrainList, ValList = percentageDivide(params.WhichExperiment.Dataset.Validation.percentage,
                                                  list(params.directories.Train.Input.Subjects),
                                                  params.WhichExperiment.Dataset.randomFlag)

            if params.WhichExperiment.Dataset.Validation.fromKeras or params.WhichExperiment.HardParams.Model.Method.Use_TestCases_For_Validation:
                DataAll.Train = separatingConcatenatingIndexes(DataAll.Train_ForTest, list(DataAll.Train_ForTest),'train')
                DataAll.Validation = ''
            else:
                DataAll.Train = separatingConcatenatingIndexes(DataAll.Train_ForTest, TrainList, 'train')
                DataAll.Validation = separatingConcatenatingIndexes(DataAll.Train_ForTest, ValList, 'validation')

            return DataAll

        def readingAllSubjects(Subjects, mode):

            def ErrorInPaddingCheck(subject):
                ErrorFlag = False
                if np.min(subject.Padding) < 0:
                    if np.min(subject.Padding) < -params.WhichExperiment.HardParams.Model.paddingErrorPatience:
                        AA = subject.address.split(os.path.dirname(subject.address) + '/')
                        shutil.move(subject.address, os.path.dirname(subject.address) + '/' + 'ERROR_' + AA[1])
                        print('WARNING: subject: ', subject.subjectName,
                              ' size is out of the training network input dimensions')
                        ErrorFlag = True
                    else:
                        Dirsave = smallFuncs.mkDir(params.directories.Test.Result + '/' + subject.subjectName, )
                        np.savetxt(Dirsave + '/paddingError.txt', subject.Padding, fmt='%d')
                        print('WARNING: subject: ', subject.subjectName, ' padding error patience activated, Error:',
                              np.min(subject.Padding))
                return ErrorFlag

            Data = {}
            for nameSubject, subject in tqdm(Subjects.items(), desc='Loading ' + mode):

                if ErrorInPaddingCheck(subject): continue

                im, imF = readingImage(params, subject)
                origMsk, msk = readingNuclei(params, subject, imF.shape)

                msk = msk > Th
                origMsk = origMsk > Th

                if im[..., 0].shape == msk[..., 0].shape:
                    Data[nameSubject] = testCase(Image=im, Mask=msk, OrigMask=(origMsk).astype('float32'),
                                                 Affine=imF.get_affine(), Header=imF.get_header(),
                                                 original_Shape=imF.shape)
                else:
                    Error_MisMatch_In_Dim_ImageMask(subject, mode, nameSubject)

            return Data

        def readValidation(DataAll):
            if params.WhichExperiment.HardParams.Model.Method.Use_TestCases_For_Validation:
                Val_Indexes = list(DataAll.Test)
                DataAll.Validation = separatingConcatenatingIndexes(DataAll.Test, Val_Indexes, 'validation')
            return DataAll

        DataAll = data()

        # Loading train subjects
        if trainFlag():
            DataAll.Train_ForTest = readingAllSubjects(params.directories.Train.Input.Subjects, 'train')

            # Separating training & validation data
            DataAll = separateTrainVal_and_concatenateTrain(DataAll)

        # Loading test subjects
        if params.directories.Test.Input.Subjects:
            DataAll.Test = readingAllSubjects(params.directories.Test.Input.Subjects, 'test')
            DataAll = readValidation(DataAll)

        # These data are used to segment the whole thalamus mask for saagittal network while running the coronal network
        
        # Loading train subjects re-oriented for sagittal network
        if sagittalFlag() and trainFlag():
            DataAll.Sagittal_Train_ForTest = readingAllSubjects(params.directories.Train.Input_Sagittal.Subjects, 'trainS')

        # Loading test subjects re-oriented for sagittal network
        if sagittalFlag():
            DataAll.Sagittal_Test = readingAllSubjects(params.directories.Test.Input_Sagittal.Subjects, 'testS')

        return DataAll

    params = preAnalysis(params)
    smallFuncs.mkDir(params.directories.Test.Result)
    Data = main_ReadingDataset(params)

    return Data, params


def percentageDivide(percentage, subjectsList, randomFlag):
    """ 
    Randomly dividing the input subjects into train & validation based on the assigned percentage 

    Args:
        percentage (float): Percentage of data to be assigned as validation 
        subjectsList (list): list of subjects
        randomFlag (boolean): True: randomizing the dataa before data division

    Returns:
        Train_List (list):   list of train subjects
        TestVal_List (list): list of validation subjects
    """    

    # Number of subjects
    L = len(subjectsList)
    indexes = np.array(range(L))

    # Shuffling the list of subjects
    if randomFlag: 
        shuffle(indexes)


    per = int(percentage * L)
    if per == 0 and L > 1: 
        per = 1

    # list of validation subjects
    TestVal_List = [subjectsList[i] for i in indexes[:per]]

    # list of training subjects
    Train_List = [subjectsList[i] for i in indexes[per:]]

    return Train_List, TestVal_List


def preAnalysis(params):
    """
    Analysis of all input subjects dimensions to find the 
      - Update number of layers
      - Amount of padding required for each subject

    Args:
        params: User parameters

    Returns:
        params: Updated user parameters
    """    

    slicingDim = params.WhichExperiment.Dataset.slicingInfo.slicingDim
    def find_AllInputSizes(params):

        def newCropedSize(subject, params, mode):

            def readingCascadeCropSizes(subject):

                if 'train' in mode:
                    dirr = params.directories.Test.Result + '/TrainData_Output/' + subject.subjectName + '/'

                elif 'test' in mode:
                    dirr = subject.address + '/' + params.UserInfo['thalamic_side'].active_side + '/sd' + str(
                        slicingDim) + '/'

                BBf = np.loadtxt(dirr + '/BB_' + params.WhichExperiment.HardParams.Model.Method.ReferenceMask + '.txt',
                                 dtype=int)
                BB = BBf[:, :2]
                BBd = BBf[:, 2:]

                # Because on the slicing direction we don't want the extra dilated effect to be considered
                BBd[slicingDim] = BB[slicingDim]
                BBd = BBd[params.WhichExperiment.Dataset.slicingInfo.slicingOrder]
                return BBd

            if 1 not in params.WhichExperiment.Nucleus.Index:

                # Corresponding to the 2nd network in the cascade framework. Segmentation of nuclei
                BB = readingCascadeCropSizes(subject)

                origSize = np.array(nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz').shape)

                # re-orienting the input slices into axial, sagittal, and coronal depending on the appropriate network
                origSize = origSize[params.WhichExperiment.Dataset.slicingInfo.slicingOrder]

                subject.NewCropInfo.OriginalBoundingBox = BB

                # The amount of cropping required to remove the extra padded pixels
                subject.NewCropInfo.PadSizeBackToOrig = tuple(
                    [tuple([BB[d][0], origSize[d] - BB[d][1]]) for d in range(3)])

                Shape = np.array([BB[d][1] - BB[d][0] for d in range(3)])

            else:

                # corresponding to the 1st network in the cascade framework. Segmentation of whole thalamus
                Shape = np.array(nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz').shape)
                Shape = Shape[params.WhichExperiment.Dataset.slicingInfo.slicingOrder]

            return Shape, subject

        def loopOverAllSubjects(Input, mode):
            inputSize = []
            for sj in Input.Subjects:
                Shape, Input.Subjects[sj] = newCropedSize(Input.Subjects[sj], params, mode)
                inputSize.append(Shape)

            Input.inputSizes = np.array(inputSize)
            return Input

        if params.directories.Train.Input.Subjects: params.directories.Train.Input = loopOverAllSubjects(
            params.directories.Train.Input, 'train')
        if params.directories.Test.Input.Subjects: params.directories.Test.Input = loopOverAllSubjects(
            params.directories.Test.Input, 'test')

        if params.WhichExperiment.Nucleus.Index[0] == 1 and slicingDim == 2:
            if params.directories.Train.Input_Sagittal.Subjects: params.directories.Train.Input_Sagittal = loopOverAllSubjects(
                params.directories.Train.Input_Sagittal, 'train')
            if params.directories.Test.Input_Sagittal.Subjects:  params.directories.Test.Input_Sagittal = loopOverAllSubjects(
                params.directories.Test.Input_Sagittal, 'test')

        return params

    def find_correctNumLayers(params):
        """
        Checking the maximum number of layers in the U-Net based on the network input dimention 

        Args:
            params: User parameters

        Returns:
            params: Updated parameters that include the updated number of layers in the U-Net
        """

        HardParams = params.WhichExperiment.HardParams

        def func_MinInputSize():
            """
            Finding the network input dimention based on all train & test inputs

            Args:
                params: User parameters

            Returns:
                MinInputSize: Network input dimension
            """  

            if params.WhichExperiment.Dataset.InputPadding.Automatic:
                inputSizes = params.directories.Test.Input.inputSizes if params.WhichExperiment.TestOnly else np.concatenate(
                    (params.directories.Train.Input.inputSizes, params.directories.Test.Input.inputSizes), axis=0)

                return np.min(inputSizes, axis=0)
            else:
                return params.WhichExperiment.Dataset.InputPadding.HardDimensions

        MinInputSize = func_MinInputSize()

        kernel_size = HardParams.Model.Layer_Params.ConvLayer.Kernel_size.conv
        num_Layers = HardParams.Model.num_Layers

        params.WhichExperiment.HardParams.Model.num_Layers_changed = False
        dim = HardParams.Model.Method.InputImage2Dvs3D

        # Check if the figure map size at the most bottom layer is bigger than convolution kernel size                
        if np.min(MinInputSize[:dim] - np.multiply(kernel_size, (2 ** (num_Layers - 1)))) < 0:
            params.WhichExperiment.HardParams.Model.num_Layers = int(
                np.floor(np.log2(np.min(np.divide(MinInputSize[:dim], kernel_size))) + 1))
            print('WARNING: INPUT IMAGE SIZE IS TOO SMALL FOR THE NUMBER OF LAYERS')
            print('# LAYERS  OLD:', num_Layers, ' =>  NEW:', params.WhichExperiment.HardParams.Model.num_Layers)
            params.WhichExperiment.HardParams.Model.num_Layers_changed = True

        return params

    def find_PaddingValues(params):
        """
        Finding the amount of padding needed for each subject based on the calculated network's input dimention

        Args:
            params : Updated parameters that includes the amount of padding required for each of train & test subjects
        """        

        def findingPaddedInputSize(params):
            inputSizes = params.directories.Test.Input.inputSizes if params.WhichExperiment.TestOnly else np.concatenate(
                (params.directories.Train.Input.inputSizes, params.directories.Test.Input.inputSizes), axis=0)
            # inputSizes = np.concatenate((params.directories.Train.Input.inputSizes , params.directories.Test.Input.inputSizes),axis=0)  

            num_Layers = params.WhichExperiment.HardParams.Model.num_Layers
            L = num_Layers - 1

            a = 2 ** (L)
            return [int(a * np.ceil(s / a)) if s % a != 0 else s for s in np.max(inputSizes, axis=0)]

        def findingSubjectsFinalPaddingAmount(wFolder, Input, params):

            def applyingPaddingDimOnSubjects(params, Input):
                fullpadding = params.WhichExperiment.HardParams.Model.InputDimensions - Input.inputSizes
                md = np.mod(fullpadding, 2)
                for sn, name in enumerate(list(Input.Subjects)):
                    padding = [tuple([0, 0])] * 4

                    for dim in range(
                            params.WhichExperiment.HardParams.Model.Method.InputImage2Dvs3D):  # params.WhichExperiment.Dataset.slicingInfo.slicingOrder[:2]:
                        if md[sn, dim] == 0:
                            padding[dim] = tuple([int(fullpadding[sn, dim] / 2)] * 2)
                        else:
                            padding[dim] = tuple(
                                [int(np.floor(fullpadding[sn, dim] / 2) + 1), int(np.floor(fullpadding[sn, dim] / 2))])

                    if np.min(tuple(padding)) < 0:
                        print('---')
                    Input.Subjects[name].Padding = tuple(padding)

                return Input

            return applyingPaddingDimOnSubjects(params, Input)

        AA = findingPaddedInputSize(
            params) if params.WhichExperiment.Dataset.InputPadding.Automatic else params.WhichExperiment.Dataset.InputPadding.HardDimensions
        params.WhichExperiment.HardParams.Model.InputDimensions = AA

        if params.directories.Train.Input.Subjects: params.directories.Train.Input = findingSubjectsFinalPaddingAmount(
            'Train', params.directories.Train.Input, params)
        if params.directories.Test.Input.Subjects:  params.directories.Test.Input = findingSubjectsFinalPaddingAmount(
            'Test', params.directories.Test.Input, params)

        if params.WhichExperiment.Nucleus.Index[0] == 1 and slicingDim == 2:
            if params.directories.Train.Input_Sagittal.Subjects: params.directories.Train.Input_Sagittal = findingSubjectsFinalPaddingAmount(
                'Train', params.directories.Train.Input_Sagittal, params)
            if params.directories.Test.Input_Sagittal.Subjects:  params.directories.Test.Input_Sagittal = findingSubjectsFinalPaddingAmount(
                'Test', params.directories.Test.Input_Sagittal, params)

        return params

    params = find_AllInputSizes(params)
    params = find_correctNumLayers(params)
    params = find_PaddingValues(params)

    return params
