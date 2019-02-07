import numpy as np
from imageio import imread
from random import shuffle
from tqdm import tqdm, trange
import nibabel as nib
import shutil 
import os, sys
from otherFuncs import smallFuncs
from preprocess import normalizeA, applyPreprocess
import matplotlib.pyplot as plt

class ImageLabel:
    Image = np.zeros(3)
    Label = ''

class info:
    Height = ''
    Width = ''

class data:
    Train = ImageLabel()
    Train_ForTest = ""
    Test = ""
    Validation = ImageLabel()
    Info = info

class trainCase:
    def __init__(self, Image, Mask):
        self.Image = Image
        self.Mask  = Mask

class testCase:
    def __init__(self, Image, Mask, OrigMask, Affine, Header, original_Shape):
        self.Image = Image
        self.Mask = Mask
        self.OrigMask  = OrigMask
        self.Affine = Affine
        self.Header = Header
        self.original_Shape = original_Shape


def check_Dataset_ForTraining(params, flag, Info):  #  mode = 'experiments' # single_run'

    if flag:
        mode = 'experiment'

        # #! copying the dataset into the experiment folder
        # if params.WhichExperiment.Dataset.CreatingTheExperiment: movingFromDatasetToExperiments(params)


        # #! preprocessing the data
        # if params.preprocess.Mode:
        #     applyPreprocess.main(params, mode)
             
        # params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)

        #! correcting the number of layers
        num_Layers = smallFuncs.correctNumLayers(params)
        params.WhichExperiment.HardParams.Model.num_Layers = num_Layers

        #! Finding the final image sizes after padding & amount of padding
        Subjects_Train, Subjects_Test, new_inputSize = smallFuncs.imageSizesAfterPadding(params, mode)

        params.directories.Train.Input.Subjects = Subjects_Train
        params.directories.Test.Input.Subjects  = Subjects_Test
        params.WhichExperiment.HardParams.Model.InputDimensions = new_inputSize


        # params.preprocess.TestOnly = True
        #! loading the dataset
        Data = loadDataset(params)
        params.WhichExperiment.HardParams.Model.imageInfo = Data.Info

        Info['num_Layers']     = num_Layers
        Info['new_inputSize']  = new_inputSize
        Info['Subjects_Train'] = Subjects_Train
        Info['Subjects_Test']  = Subjects_Test
        Info['imageInfo']      = Data.Info

        return Data, params, Info

    else:
        params.WhichExperiment.HardParams.Model.num_Layers      = Info['num_Layers']
        params.WhichExperiment.HardParams.Model.imageInfo       = Info['imageInfo']
        params.directories.Train.Input.Subjects                 = Info['Subjects_Train']
        params.directories.Test.Input.Subjects                  = Info['Subjects_Test']
        params.WhichExperiment.HardParams.Model.InputDimensions = Info['new_inputSize']

        return '', params, ''

def one_hot(a, num_classes):
  return np.eye(num_classes)[a]

def DatasetsInfo(DatasetIx):
    switcher = {
        1: ('SRI_3T', '/array/ssd/msmajdi/data/preProcessed/3T/SRI_3T'),
        2: ('kaggleCompetition', '/array/ssd/msmajdi/data/originals/KaggleCompetition/train'),
        3: ('fashionMnist', 'intrinsic'),
        4: ('All_7T', '/array/ssd/msmajdi/data/preProcessed/7T/All_7T'),
        5: ('20priors', '/array/ssd/msmajdi/data/preProcessed/7T/20priors'),
    }
    return switcher.get(DatasetIx, 'WARNING: Invalid dataset index')

def loadDataset(params):

    if 'fashionMnist' in params.WhichExperiment.Dataset.name:
        Data, _ = fashionMnist(params)
    elif 'kaggleCompetition' in params.WhichExperiment.Dataset.name:
        Data, _ = kaggleCompetition(params)
    else:
        # Data = readingFromExperiments3D(params)
        Data = readingFromExperiments3D_new(params)

    _, Data.Info.Height, Data.Info.Width, _ = Data.Test[list(Data.Test)[0]].Image.shape if params.preprocess.TestOnly else Data.Train.Image.shape
    # params.WhichExperiment.HardParams.Model.imageInfo = Data.Info
    return Data

def fashionMnist(params):
    from keras.datasets import fashion_mnist
    
    fullData  = fashion_mnist.load_data()

    images = (np.expand_dims(fullData[0][0],axis=3)).astype('float32') / 255
    masks  = one_hot(fullData[0][1],10)

    if params.WhichExperiment.Dataset.Validation.fromKeras:
        data.Train.Image = images
        data.Train.Mask = masks
    else:
        data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage, images, masks, params.WhichExperiment.Dataset.randomFlag)

    data.Test.Image = (np.expand_dims(fullData[1][0],axis=3)).astype('float32') / 255
    data.Test.Label = one_hot(fullData[1][1],10)
    return data, '_'

def kaggleCompetition(params):

    subF = next(os.walk(params.WhichExperiment.Dataset.address))

    for ind in trange(min(len(subF[1]),50), desc='Loading Dataset'):

        imDir = subF[0] + '/' + subF[1][ind] + '/images'
        imMsk = subF[0] + '/' + subF[1][ind] + '/masks'
        a = next(os.walk(imMsk))
        b = next(os.walk(imDir))

        im = np.squeeze(imread(imDir + '/' + b[2][0])[:256,:256,0])
        # im = ndimage.zoom(im,(1,1,2),order=3)

        im = np.expand_dims(im,axis=0)
        im = (np.expand_dims(im,axis=3)).astype('float32') / 255
        images = im if ind == 0 else np.concatenate((images,im),axis=0)

        msk = imread(imMsk + '/' + a[2][0]) if len(a[2]) > 0 else np.zeros(im.shape)
        if len(a[2]) > 1:
            for sF in range(1,len(a[2])):
                msk = msk + imread(imMsk + '/' + a[2][sF])

        msk = np.expand_dims(msk[:256,:256],axis=0)
        msk = (np.expand_dims(msk,axis=3)).astype('float32') / 255
        masks = msk>0.5 if ind == 0 else np.concatenate((masks,msk>0.5 ),axis=0)

    masks = np.concatenate((masks,1-masks),axis=3)

    if params.WhichExperiment.Dataset.Validation.fromKeras:
        data.Train.Image = images
        data.Train.Mask = masks
    else:
        data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage ,images, masks, params.WhichExperiment.Dataset.randomFlag)

    data.Test = data.Train
    return data, '_'

def inputPreparationForUnet(im,subject, params):

    im = np.transpose(im, params.WhichExperiment.Dataset.slicingInfo.slicingOrder)
    im = np.pad(im, subject.Padding[:3], 'constant')
    im = np.transpose(im,[2,0,1])
    im = np.expand_dims(im ,axis=3).astype('float32')
    return im

def backgroundDetector(masks):
    a = np.sum(masks,axis=3)
    background = np.zeros(masks.shape[:3])
    background[np.where(a == 0)] = 1
    background = np.expand_dims(background,axis=3)
    return background

# TODO: also I need to finish this function
# TODO: add the saving images with the format mahesh said
# TODO: maybe add the ability to crop the test cases with bigger sizes than network input dimention accuired from train datas
def readingFromExperiments3D_new(params):


    def readingImage(params, subject):
        imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
        im = inputPreparationForUnet(imF.get_data(), subject, params)
        im = normalizeA.main_normalize(params.preprocess.Normalize , im)
        return im, imF

    def readingNuclei(params, subject, imFshape):
        for cnt, NucInd in enumerate(params.WhichExperiment.Nucleus.Index):
            nameNuclei, _ = smallFuncs.NucleiSelection(NucInd)
            inputMsk = subject.Label.address + '/' + nameNuclei + '_PProcessed.nii.gz'

            origMsk1N = nib.load(inputMsk).get_data() if os.path.exists(inputMsk) else np.zeros(imFshape) 
            msk1N = inputPreparationForUnet(origMsk1N, subject, params)
            origMsk1N = np.expand_dims(origMsk1N ,axis=3)

            origMsk = origMsk1N if cnt == 0 else np.concatenate((origMsk, origMsk1N) ,axis=3).astype('float32')
            msk = msk1N if cnt == 0 else np.concatenate((msk,msk1N),axis=3).astype('float32')

        background = backgroundDetector(msk)
        msk = np.concatenate((msk, background),axis=3).astype('float32')
        
        return origMsk , msk
            

    def Error_In_Dimention(cntSkipped,subject, mode, nameSubject):
        AA = subject.address.split('vimp')
        shutil.move(subject.address, AA[0] + 'ERROR_vimp' + AA[1])
        print('WARNING:', mode , cntSkipped + 1 , nameSubject, ' image and mask have different shape sizes')
        return cntSkipped + 1

    TestData, TrainData = {}, {}
    for mode in ['train','test']:

        if 'train' in mode and params.preprocess.TestOnly:
            continue

        Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects

        #! reading all images and concatenating them into one array
        Th = 0.5*params.WhichExperiment.HardParams.Model.LabelMaxValue
        cntSkipped = 0
        indTrain = 0
        for _, nameSubject in tqdm(enumerate(Subjects), desc='Loading Dataset: ' + mode):
            subject = Subjects[nameSubject]

            # TODO: replace this with cropping if the negative number is low e.g. less than 5
            if np.min(subject.Padding) < 0:
                print('WARNING: subject: ',nameSubject,' size is out of the training network input dimensions')
                continue

            im, imF = readingImage(params, subject)
            # imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
            # im = inputPreparationForUnet(imF.get_data(), subject, params)
            # im = normalizeA.main_normalize(params.preprocess.Normalize , im)


            origMsk , msk = readingNuclei(params, subject, imF.shape)
            # for cnt, NucInd in enumerate(params.WhichExperiment.Nucleus.Index):
            #     nameNuclei, _ = smallFuncs.NucleiSelection(NucInd)
            #     inputMsk = subject.Label.address + '/' + nameNuclei + '_PProcessed.nii.gz'
            #     origMsk1N = nib.load(inputMsk).get_data() if os.path.exists(inputMsk) else np.zeros(imF.shape)
            #     msk1N = inputPreparationForUnet(origMsk1N, subject, params)
            #     origMsk1N = np.expand_dims(origMsk1N ,axis=3)
            #     origMsk = origMsk1N if cnt == 0 else np.concatenate((origMsk, origMsk1N) ,axis=3).astype('float32')
            #     msk = msk1N if cnt == 0 else np.concatenate((msk,msk1N),axis=3).astype('float32')
            # background = backgroundDetector(msk)
            # msk = np.concatenate((msk, background),axis=3).astype('float32')

            if 'ERROR_vimp' not in nameSubject:
                if im[...,0].shape == msk[...,0].shape:
                    if 'train' in mode:                        
                        images = im     if indTrain == 0 else np.concatenate((images,im    ),axis=0)
                        masks  = msk>Th if indTrain == 0 else np.concatenate((masks,msk>Th ),axis=0)
                        TrainData[nameSubject] = testCase(Image=im, Mask=msk ,OrigMask=origMsk.astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)
                        indTrain = indTrain + 1

                    elif 'test' in mode:
                        TestData[nameSubject]  = testCase(Image=im, Mask=msk ,OrigMask=origMsk.astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)
                else:
                    cntSkipped = Error_In_Dimention(cntSkipped,subject, mode, nameSubject)
                    # cntSkipped = cntSkipped + 1
                    # AA = subject.address.split('vimp')
                    # shutil.move(subject.address, AA[0] + 'ERROR_vimp' + AA[1])
                    # print('WARNING:', mode , cntSkipped , nameSubject, ' image and mask have different shape sizes')

        if 'train' in mode:
            data.Train_ForTest = TrainData

            if params.WhichExperiment.Dataset.Validation.fromKeras:
                data.Train = trainCase(Image=images, Mask=masks.astype('float32'))
            else:
                data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage, images, masks, params.WhichExperiment.Dataset.randomFlag)
        else:
            data.Test = TestData
            
    return data

def TrainValSeperate(percentage, images, masks, randomFlag):

    subjectsList = list(range(images.shape[0]))
    TrainList, ValList = percentageDivide(percentage, subjectsList, randomFlag)


    Validation = ImageLabel()
    Validation.Image = images[ValList, ...]
    Validation.Label = masks[ValList, ...]

    Train = ImageLabel()
    Train.Image = images[TrainList, ...]
    Train.Label = masks[TrainList, ...]

    return Train, Validation

def percentageDivide(percentage, subjectsList, randomFlag):

    L = len(subjectsList)
    indexes = np.array(range(L))

    if randomFlag: shuffle(indexes)
    per = int( percentage * L )
    if per == 0 and L > 1: per = 1

    # TestValList = subjectsList[indexes[:per]]
    TestValList = [subjectsList[i] for i in indexes[:per]]
    # TrainList = subjectsList[indexes[per:]]
    TrainList = [subjectsList[i] for i in indexes[per:]]

    return TrainList, TestValList

def movingFromDatasetToExperiments(params):

    def checkAugmentedData(params):

        def listAugmentationFolders(mode):
            Dir_Aug1 = params.WhichExperiment.Dataset.address + '/Augments/' + mode
            flag_Aug = os.path.exists(Dir_Aug1)

            ListAugments = smallFuncs.listSubFolders(Dir_Aug1) if flag_Aug else list('')

            return flag_Aug, {'address': Dir_Aug1 , 'list': ListAugments , 'mode':mode}

        flagAg, AugDataL = np.zeros(3), list(np.zeros(3))
        if params.Augment.Mode:
            if params.Augment.Linear.Rotation.Mode:     flagAg[0], AugDataL[0] = listAugmentationFolders('Linear_Rotation')
            if params.Augment.Linear.Shift.Mode:        flagAg[1], AugDataL[1] = listAugmentationFolders('Linear_Shift')
            if params.Augment.NonLinear.Mode: flagAg[2], AugDataL[2] = listAugmentationFolders('NonLinear')

        return flagAg, AugDataL
        
    def copyAugmentData(DirOut, AugDataL, subject):
        if 'NonLinear' in AugDataL['mode']: AugDataL['list'] = [i for i in AugDataL['list'] if subject in i.split('Ref_')[0]]

        for subjectsAgm in AugDataL['list']:
            if subject in subjectsAgm: shutil.copytree(AugDataL['address'] + '/' + subjectsAgm  ,  DirOut + '/' + subjectsAgm)
                    
    if len(os.listdir(params.directories.Train.address)) != 0 or len(os.listdir(params.directories.Test.address)) != 0:
        print('*** DATASET ALREADY EXIST; PLEASE REMOVE \'train\' & \'test\' SUBFOLDERS ***')
        sys.exit
    
    else:
        List = smallFuncs.listSubFolders(params.WhichExperiment.Dataset.address)
        flagAg, AugDataL = checkAugmentedData(params)

        TestParams  = params.WhichExperiment.Dataset.Test
        _, TestList = percentageDivide(TestParams.percentage, List, params.WhichExperiment.Dataset.randomFlag) if 'percentage' in TestParams.mode else TestParams.subjects
        for subject in List:

            DirOut, mode = (params.directories.Test.address , 'test') if subject in TestList else (params.directories.Train.address, 'train')

            if not os.path.exists(DirOut + '/' + subject):
                shutil.copytree(params.WhichExperiment.Dataset.address + '/' + subject  ,  DirOut + '/' + subject)

            if 'train' in mode:
                for AgIx in range(len(AugDataL)):
                    if flagAg[AgIx]: copyAugmentData(DirOut, AugDataL[AgIx], subject)

        params = smallFuncs.inputNamesCheck(params, 'experiment')
        
    return True
