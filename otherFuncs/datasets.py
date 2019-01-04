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


def one_hot(a, num_classes):
  return np.eye(num_classes)[a]

class ImageLabel:
    Image = np.zeros(3)
    Label = ''

class info:
    Height = ''
    Width = ''

class data:
    Train = ImageLabel()
    Test = ImageLabel()
    Validation = ImageLabel()
    Info = info

def loadDataset(params):

    if 'fashionMnist' in params.WhichExperiment.Dataset.name:
        Data, _ = fashionMnist(params)
    elif 'kaggleCompetition' in params.WhichExperiment.Dataset.name:
        Data, _ = kaggleCompetition(params)
    elif 'SRI_3T' in params.WhichExperiment.Dataset.name:
        Data = readingFromExperiments(params)
        Data.Train.Image = normalizeA.main_normalize(params.preprocess.Normalize , Data.Train.Image)
        Data.Test.Image  = normalizeA.main_normalize(params.preprocess.Normalize , Data.Test.Image)

    _, Data.Info.Height, Data.Info.Width, _ = Data.Train.Image.shape

    return Data

def fashionMnist(params):
    fullData  = fashion_mnist.load_data()

    images = (np.expand_dims(fullData[0][0],axis=3)).astype('float32') / 255
    masks  = one_hot(fullData[0][1],10)

    if params.WhichExperiment.Dataset.Validation.fromKeras:
        data.Train.Image = images
        data.Train.Label = masks
    else:
        data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage, images, masks)

    data.Test.Image = (np.expand_dims(fullData[1][0],axis=3)).astype('float32') / 255
    data.Test.Label = one_hot(fullData[1][1],10)
    return data, '_'

def kaggleCompetition(params):

    dir = '/array/ssd/msmajdi/data/original/KaggleCompetition/train'
    subF = next(os.walk(dir))

    for ind in trange(min(len(subF[1]),5), desc='Loading Dataset'):

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
        data.Train.Label = masks
    else:
        data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage ,images, masks)

    data.Test = data.Train
    return data, '_'

# TODO: also I need to finish this function
# TODO: add the saving images with the format mahesh said
# TODO: maybe add the ability to crop the test cases with bigger sizes than network input dimention accuired from train datas
def readingFromExperiments(params):

    for mode in ['train','test']:

        if params.preprocess.TestOnly and 'train' in mode:
            continue

        Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects

        #! reading all images and concatenating them into one array
        Th = 0.5*params.WhichExperiment.HardParams.Model.LabelMaxValue
        for ind, name in tqdm(enumerate(Subjects), desc='Loading Dataset'):
            subject = Subjects[name]

            if np.min(subject.Padding) < 0:
                print('WARNING: subject: ',name,' size is out of the training network input dimensions')
                continue

            im = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz').get_data()
            im = np.pad(im, subject.Padding, 'constant')
            im = np.transpose(im,[2,0,1])

            if os.path.exists(subject.Label.address + '/' + subject.Label.LabelProcessed + '.nii.gz'):
                msk = nib.load(subject.Label.address + '/' + subject.Label.LabelProcessed + '.nii.gz').get_data()
                msk = np.pad(msk, subject.Padding, 'constant')
                msk = np.transpose(msk,[2,0,1])
            else:
                msk = np.zeros(im.shape)

            images = im     if ind == 0 else np.concatenate((images,im    ),axis=0)
            masks  = msk>Th if ind == 0 else np.concatenate((masks,msk>Th ),axis=0)
        masks  = np.expand_dims(masks,axis=3)
        images = np.expand_dims(images,         axis=3).astype('float32')
        masks  = np.concatenate((masks,1-masks),axis=3).astype('float32')

        if 'train' in mode:
            if params.WhichExperiment.Dataset.Validation.fromKeras:
                data.Train.Image = images
                data.Train.Label = masks
            else:
                data.Train, data.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage, images, masks)
        else:
            data.Test.Image = images
            data.Test.Label = masks

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
