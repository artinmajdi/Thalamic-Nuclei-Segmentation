import numpy as np
from imageio import imread
from random import shuffle
from tqdm import tqdm, trange
import nibabel as nib
import collections
import os, sys, collections
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras.datasets import fashion_mnist, mnist

from otherFuncs import params

from preprocess.normalizeMine import normalizeMain
from otherFuncs.smallFuncs import imageSizesAfterPadding

def one_hot(a, num_classes):
  return np.eye(num_classes)[a]

class ImageLabel:
    Image = np.zeros(3)
    Label = ''

class info:
    Height = ''
    Width = ''

class Data:
    Train = ImageLabel()
    Test = ImageLabel()
    Validation = ImageLabel()
    Info = info

def loadDataset(params):

    ModelParam = params.directories.WhichExperiment.HardParams.Model
    if 'fashionMnist' in ModelParam.dataset:
        Data, _ = fashionMnist(ModelParam)

    elif 'kaggleCompetition' in ModelParam.dataset:
        Data, _ = kaggleCompetition(ModelParam)

    elif 'SRI_3T' in ModelParam.dataset:
        Data, params = readingFromExperiments(params)
        Data.Train.Image = normalizeMain(params.preprocess.Normalize , Data.Train.Image)
        Data.Test.Image  = normalizeMain(params.preprocess.Normalize , Data.Test.Image)

    _, Data.Info.Height, Data.Info.Width, _ = Data.Train.Image.shape

    return Data, params

def fashionMnist(ModelParam):
    data  = fashion_mnist.load_data()

    images = (np.expand_dims(data[0][0],axis=3)).astype('float32') / 255
    masks  = one_hot(data[0][1],10)

    if ModelParam.Validation.fromKeras:
        Data.Train.Image = images
        Data.Train.Label = masks
    else:
        Data.Train, Data.Validation = TrainValSeperate(ModelParam, images, masks)

    Data.Test.Image = (np.expand_dims(data[1][0],axis=3)).astype('float32') / 255
    Data.Test.Label = one_hot(data[1][1],10)
    return Data, '_'

def kaggleCompetition(ModelParam):

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

    if ModelParam.Validation.fromKeras:
        Data.Train.Image = images
        Data.Train.Label = masks
    else:
        Data.Train, Data.Validation = TrainValSeperate(ModelParam ,images, masks)

    Data.Test = Data.Train
    return Data, '_'

# TODO: I need to add the ability for test files to read the padding size from training instead of measuring it again
# TODO: also I need to finish this function
# TODO: add the saving images with the format mahesh said
# TODO: check why the label & image has different crop sizes; maybe if i rerun it it will fix it
def readingFromExperiments(params):

    # for mode in ['Train','Test']:

        # if params.preprocess.TestOnly and 'Train' in mode:
        #     continue

    Subjects = params.directories.Train.Input.Subjects
    HardParams = params.directories.WhichExperiment.HardParams

    #! Finding the final image sizes after padding & amount of padding
    Subjects, HardParams = imageSizesAfterPadding(Subjects, HardParams)


    #! reading all images and concatenating them into one array
    Th = 0.5*HardParams.Model.LabelMaxValue
    for ind, name in tqdm(enumerate(Subjects), desc='Loading Dataset'):
        subject = Subjects[name]
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

    images = np.expand_dims(images,         axis=3).astype('float32')
    masks  = np.concatenate((masks,1-masks),axis=3).astype('float32')

    if HardParams.Model.Validation.fromKeras:
        Data.Train.Image = images
        Data.Train.Label = masks
    else:
        Data.Train, Data.Validation = TrainValSeperate(HardParams.Model ,images, masks)


    params.directories.Train.Input.Subjects = Subjects
    params.directories.WhichExperiment.HardParams = HardParams


    return Data, params

def TrainValSeperate(ModelParam, images, masks):

    indexes = np.array(range(images.shape[0]))
    shuffle(indexes)
    per = int( ModelParam.Validation.percentage * images.shape[0] )
    if per == 0 and images.shape[0] > 1: per = 1
    Validation = ImageLabel()
    Validation.Image = images[indexes[:per],...]
    Validation.Label = masks[indexes[:per],...]

    Train = ImageLabel()
    Train.Image = images[indexes[per:],...]
    Train.Label = masks[indexes[per:],...]

    return Train, Validation

def movingFromDatasetToExperiments(params):
    return True
