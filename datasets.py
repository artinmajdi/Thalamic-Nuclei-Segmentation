
from keras.datasets import fashion_mnist, mnist
import numpy as np
import os
from imageio import imread

def one_hot(a, num_classes):
  return np.eye(num_classes)[a]

class Train:
    Data = ''
    Label = ''

class Test:
    Data = ''
    Label = ''

class Info:
    Height = ''
    Width = ''

def loadDataset(dataset):

    if 'fashionMnist' in dataset:
        Train, Test, Info = fashionMnist()

    elif 'kaggleCompetition' in dataset:
        Train, Test, Info = kaggleCompetition()


    _, Info.Height, Info.Width = Train.Data.shape

    return Train, Test, Info



def fashionMnist():
    data = fashion_mnist.load_data()
    Train.Data  = (np.expand_dims(data[0][0],axis=3)).astype('float32') / 255
    Train.Label = one_hot(data[0][1],10)

    Test.Data  = (np.expand_dims(data[1][0],axis=3)).astype('float32') / 255
    Test.Label = one_hot(data[1][1],10)
    return Train, Test

def kaggleCompetition():
    dir = '/array/ssd/msmajdi/data/KaggleCompetition/train'
    subF = next(os.walk(dir))

    for ind in range(min(len(subF[1]),20)):

        imDir = subF[0] + '/' + subF[1][ind] + '/images'
        imMsk = subF[0] + '/' + subF[1][ind] + '/masks'
        a = next(os.walk(imMsk))
        b = next(os.walk(imDir))

        im = np.squeeze(imread(imDir + '/' + b[2][0])[:256,:256,0])
        # im = ndimage.zoom(im,(1,1,2),order=3)

        im = np.expand_dims(im,axis=0)
        images = im if ind == 0 else np.concatenate((images,im),axis=0)

        msk = imread(imMsk + '/' + a[2][0]) if len(a[2]) > 0 else np.zeros(im.shape)
        if len(a[2]) > 1:
            for sF in range(1,len(a[2])):
                msk = msk + imread(imMsk + '/' + a[2][sF])

        msk = np.expand_dims(msk,axis=0)
        masks = msk if ind == 0 else np.concatenate((masks,msk),axis=0)


    Train.Data, Train.Label  = images, masks
    Test = Train
    return Train, Test
