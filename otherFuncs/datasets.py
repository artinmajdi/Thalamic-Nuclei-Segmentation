import numpy as np
from imageio import imread
from random import shuffle
from tqdm import tqdm, trange
import nibabel as nib
from shutil import copytree
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras.datasets import fashion_mnist
from otherFuncs import smallFuncs
from preprocess import normalizeA
from Parameters import Classes

# import h5py
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



def one_hot(a, num_classes):
  return np.eye(num_classes)[a]

def DatasetsInfo(DatasetIx):
    switcher = {
        1: ('SRI_3T', '/array/ssd/msmajdi/data/preProcessed/SRI_3T'),
        2: ('kaggleCompetition', '/array/ssd/msmajdi/data/originals/KaggleCompetition/train'),
        3: ('fashionMnist', 'intrinsic')
    }
    return switcher.get(DatasetIx, 'WARNING: Invalid dataset index')


def loadDataset(params):

    if 'fashionMnist' in params.WhichExperiment.Dataset.name:
        Data, _ = fashionMnist(params)
    elif 'kaggleCompetition' in params.WhichExperiment.Dataset.name:
        Data, _ = kaggleCompetition(params)
    elif 'SRI_3T' in params.WhichExperiment.Dataset.name:
        # Data = readingFromExperiments3D(params)
        Data = readingFromExperiments3D_new(params)

    _, Data.Info.Height, Data.Info.Width, _ = Data.Test[list(Data.Test)[0]].Image.shape if params.preprocess.TestOnly else Data.Train.Image.shape
    # params.WhichExperiment.HardParams.Model.imageInfo = Data.Info
    return Data


def fashionMnist(params):
    fullData  = fashion_mnist.load_data()

    images = (np.expand_dims(fullData[0][0],axis=3)).astype('float32') / 255
    masks  = one_hot(fullData[0][1],10)

    if params.WhichExperiment.Dataset.Validation.fromKeras:
        data.Train.Image = images
        data.Train.Mask = masks
    else:
        data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage, images, masks)

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
        data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage ,images, masks)

    data.Test = data.Train
    return data, '_'

def inputPreparationForUnet(im,subject):
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


def readingFromExperiments3D_new(params):

    TestData = {}
    TrainData = {}

    for mode in ['train','test']:

        if 'train' in mode and params.preprocess.TestOnly:
            continue

        Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects

        #! reading all images and concatenating them into one array
        Th = 0.5*params.WhichExperiment.HardParams.Model.LabelMaxValue
        for ind, nameSubject in tqdm(enumerate(Subjects), desc='Loading Dataset'):
            subject = Subjects[nameSubject]

            # TODO: replace this with cropping if the negative number is low e.g. less than 5
            if np.min(subject.Padding) < 0:
                print('WARNING: subject: ',nameSubject,' size is out of the training network input dimensions')
                continue

            imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
            im = inputPreparationForUnet(imF.get_data(), subject)
            im = normalizeA.main_normalize(params.preprocess.Normalize , im)


            for cnt, NucInd in enumerate(params.WhichExperiment.Nucleus.Index):
                nameNuclei, _ = smallFuncs.NucleiSelection(NucInd)
                inputMsk = subject.Label.address + '/' + nameNuclei + '_PProcessed.nii.gz'

                origMsk1N = nib.load(inputMsk).get_data() if os.path.exists(inputMsk) else np.zeros(imF.shape)
                origMsk = np.expand_dims(origMsk1N ,axis=3) if cnt == 0 else np.concatenate((origMsk, np.expand_dims(origMsk1N ,axis=3)) ,axis=3).astype('float32')

                msk1N = inputPreparationForUnet(origMsk1N, subject)
                msk = msk1N if cnt == 0 else np.concatenate((msk,msk1N),axis=3).astype('float32')

            background = backgroundDetector(msk)
            msk = np.concatenate((msk, background),axis=3).astype('float32')

            if 'train' in mode:
                images = im     if ind == 0 else np.concatenate((images,im    ),axis=0)
                masks  = msk>Th if ind == 0 else np.concatenate((masks,msk>Th ),axis=0)
                TrainData[nameSubject] = Classes.testCase(Image=im, Mask=msk ,OrigMask=origMsk.astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)
            elif 'test' in mode:
                TestData[nameSubject]  = Classes.testCase(Image=im, Mask=msk ,OrigMask=origMsk.astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)

        if 'train' in mode:
            data.Train_ForTest = TrainData

            if params.WhichExperiment.Dataset.Validation.fromKeras:
                data.Train = Classes.trainCase(Image=images, Mask=masks.astype('float32'))
            else:
                data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage, images, masks)
        else:
            data.Test = TestData


    return data

# TODO: also I need to finish this function
# TODO: add the saving images with the format mahesh said
# TODO: maybe add the ability to crop the test cases with bigger sizes than network input dimention accuired from train datas
def readingFromExperiments3D(params):



    TestData = {}
    TrainData = {}
    for mode in ['train','test']:

        if 'train' in mode and params.preprocess.TestOnly:
            continue

        Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects

        #! reading all images and concatenating them into one array
        Th = 0.5*params.WhichExperiment.HardParams.Model.LabelMaxValue
        for ind, name in tqdm(enumerate(Subjects), desc='Loading Dataset'):
            subject = Subjects[name]

            # TODO: replace this with cropping if the negative number is low e.g. less than 5
            if np.min(subject.Padding) < 0:
                print('WARNING: subject: ',name,' size is out of the training network input dimensions')
                continue

            imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')


            if os.path.exists(subject.Label.address + '/' + subject.Label.LabelProcessed + '.nii.gz'):
                OrigMsk = nib.load(subject.Label.address + '/' + subject.Label.LabelProcessed + '.nii.gz').get_data()
            else:
                OrigMsk = np.zeros(imF.shape)

            # a = np.mean(np.mean(OrigMsk,axis=1),axis=0)
            # b = np.where(a != 0)[0]

            # if 0:
            #     msk = OrigMsk[:,:,b]
            #     im = imF.get_data()[:,:,b]
            # else:
            msk = OrigMsk
            im = imF.get_data()


            im = np.pad(im, subject.Padding, 'constant')
            im = np.transpose(im,[2,0,1])
            im = np.expand_dims(im ,axis=3).astype('float32')
            im = normalizeA.main_normalize(params.preprocess.Normalize , im)


            msk = np.pad(msk, subject.Padding, 'constant')
            msk = np.transpose(msk,[2,0,1])
            msk = np.expand_dims(msk,axis=3) # .astype('float32')
            msk = np.concatenate((msk,1-msk),axis=3).astype('float32')

            if 'train' in mode:
                images = im     if ind == 0 else np.concatenate((images,im    ),axis=0)
                masks  = msk>Th if ind == 0 else np.concatenate((masks,msk>Th ),axis=0)

                TrainData[name] = Classes.testCase(Image=im, Mask=msk ,OrigMask=OrigMsk.astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)

            elif 'test' in mode:
                TestData[name] = Classes.testCase(Image=im, Mask=msk , OrigMask=OrigMsk.astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)


        if 'train' in mode:
            data.Train_ForTest = TrainData

            if params.WhichExperiment.Dataset.Validation.fromKeras:
                data.Train = Classes.trainCase(Image=images, Mask=masks.astype('float32'))
            else:
                data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage, images, masks)
        else:
            data.Test = TestData


    return data

def TrainValSeperate(percentage, images, masks):

    subjectsList = list(range(images.shape[0]))
    TrainList, ValList = percentageRandomDivide(percentage, subjectsList)


    Validation = ImageLabel()
    Validation.Image = images[ValList, ...]
    Validation.Label = masks[ValList, ...]

    Train = ImageLabel()
    Train.Image = images[TrainList, ...]
    Train.Label = masks[TrainList, ...]

    return Train, Validation

def percentageRandomDivide(percentage, subjectsList):

    L = len(subjectsList)
    indexes = np.array(range(L))
    shuffle(indexes)
    per = int( percentage * L )
    if per == 0 and L > 1: per = 1

    TestValList = subjectsList[:per]
    TrainList = subjectsList[per:]

    return TrainList, TestValList

def movingFromDatasetToExperiments(params):

    List = smallFuncs.listSubFolders(params.WhichExperiment.Dataset.address)
    TestParams = params.WhichExperiment.Dataset.Test
    _, TestList = percentageRandomDivide(TestParams.percentage, List) if 'percentage' in TestParams.mode else TestParams.subjects
    for subjects in List:
        DirOut = params.directories.Test.address if subjects in TestList else params.directories.Train.address
        if not os.path.exists(DirOut + '/' + subjects):
            copytree(params.WhichExperiment.Dataset.address + '/' + subjects  ,  DirOut + '/' + subjects)

    return True
