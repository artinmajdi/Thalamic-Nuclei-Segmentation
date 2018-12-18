from keras.datasets import fashion_mnist, mnist
import numpy as np
import os
from imageio import imread
import params
from random import shuffle

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

def loadDataset(ModelParam):

    if 'fashionMnist' in ModelParam.dataset:
        Data = fashionMnist(ModelParam)

    elif 'kaggleCompetition' in ModelParam.dataset:
        Data = kaggleCompetition(ModelParam)


    _, Data.Info.Height, Data.Info.Width, _ = Data.Train.Image.shape

    return Data

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
    return Data

def kaggleCompetition(ModelParam):
    dir = '/array/ssd/msmajdi/data/KaggleCompetition/train'
    subF = next(os.walk(dir))

    for ind in range(min(len(subF[1]),30)):

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
    return Data

def TrainValSeperate(ModelParam, images, masks):
    indexes = np.array(range(images.shape[0]))
    shuffle(indexes)
    per = int( ModelParam.Validation.percentage * images.shape[0] )

    Validation = ImageLabel()
    Validation.Image = images[indexes[:per],...]
    Validation.Label = masks[indexes[:per],...]

    Train = ImageLabel()
    Train.Image = images[indexes[per:],...]
    Train.Label = masks[indexes[per:],...]

    return Train, Validation
