import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from keras import models as kerasmodels
from keras import layers, losses, optimizers
import numpy as np
from skimage.filters import threshold_otsu
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import nibabel as nib
from scipy import ndimage
from shutil import copyfile
from skimage import measure
from tqdm import tqdm
from scipy.ndimage import zoom

class subject:
    def __init__(self, Address , Shape , Padding, Name, origShape):
        self.Address = Address
        self.Shape = Shape
        self.Padding = Padding
        self.Name = Name
        self.origShape = origShape

class DatatrainTest:
    def __init__(self, Image, Mask, Affine, Header):
        self.Image = Image
        self.Mask = Mask
        self.Affine = Affine
        self.Header = Header


def main():

    params = func_params()
    Data, params = readingData(params)
    
    model = trainModel(Data.training, params)

    Prediction = testModel(model, Data.testing, params)

def func_params():
    class imageInfo:
        Height = ''
        Width = ''
        depth = ''

    class modelparam:
        Dropout = 0.2
        ImageInfo = imageInfo
        num_Layers = 3
        FinalDimension = ''
        batch_size = 40
        epochs = 10
        optimizer = optimizers.adam()
        loss = losses.binary_crossentropy
        metrics = ['acc',mDice]
        downsampleFactor = 3

    class dataClass:
        Subjects = {}
        Address = ''

    class trainTest:
        Train = dataClass()
        Test = dataClass()

    class params:
        Input = trainTest()
        Address = '/array/ssd/msmajdi/experiments/keras/exp_cropping'
        Modelparam = modelparam

    params.Input.Train.Address = params.Address + '/train'
    params.Input.Test.Address = params.Address + '/test'

    return params

def trainModel(trainData, params):

    def Architecture(Modelparam):
        inputs = layers.Input( (Modelparam.FinalDimension[0], Modelparam.FinalDimension[1], Modelparam.FinalDimension[2], 1) )
        conv = inputs

        for nL in range(Modelparam.num_Layers -1):
            conv = layers.Conv3D(filters=64*(2**nL), kernel_size=(3,3,3), padding='SAME', activation='relu')(conv)
            conv = layers.Dropout(Modelparam.Layer_Params.Dropout)(conv)

        final  = layers.Conv3D(filters=2, kernel_size=(3,3,3), padding='SAME', activation='relu')(conv)

        model = kerasmodels.Model(inputs=[inputs], outputs=[final])

        return model

    model = Architecture(params.Modelparam)

    model.compile(optimizer=params.Modelparam.optimizer, loss=params.Modelparam.loss , metrics=params.Modelparam.metrics)

    hist = model.fit(x=trainData.Images, y=trainData.Masks, batch_size=params.Modelparam.batch_size, epochs=params.Modelparam.epochs, shuffle=True, validation_split=0.1, verbose=0) 

    smallFuncs.mkDir(params.Address + '/models')
    model.save(params.Address + '/models/model.h5', overwrite=True, include_optimizer=True )
    model.save_weights(params.Address + '/models/model_weights.h5', overwrite=True )

    return model

def testModel(model, testData, params):

    def perdict1Subject(Image,pad):                
        pred = model.predict(Image)
        pred = np.transpose(pred[...,0],[1,2,3,0])
        return pred[pad[0][0]:-pad[0][1] , pad[1][0]:-pad[1][1] , pad[2][0]:-pad[2][1]]

    Prediction = {}
    for subj in params.Input.Train.Subjects:
        Prediction[subj.Name] = perdict1Subject(testData.Train[subj.Name].Image , subj.Padding)
        
    for subj in params.Input.Test.Subjects:
        Prediction[subj.Name] = perdict1Subject(testData.Test[subj.Name].Image , subj.Padding)

    return Prediction

def mDice(y_true,y_pred):

    import tensorflow as tf
    return tf.reduce_sum(tf.multiply(y_true,y_pred))*2/( tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-5)

def readingData(params):
                             
    def preAnalysis(params):
            
        def findPadding(params):

            def findFinalDimension(params):

                InputSizes = []
                for sub in params.Input.Train.Subjects: InputSizes.append(sub.Shape)
                                    
                new_inputSize = np.max(InputSizes, axis=0)                

                a = 2**(params.Modelparam.num_Layers - 1)
                for dim in range(3):
                    if new_inputSize[dim] % a != 0: new_inputSize[dim] = a * np.ceil(new_inputSize[dim] / a)
                
                return new_inputSize
                
            def findPadding1Subject(subj):
                diff = params.Modelparam.FinalDimension - subj.Shape
                md = np.mod(diff,2)
                                    
                padding = [tuple([0,0])]*4
                for dim in range(3): 
                    if md[dim] == 0:
                        padding[dim] = tuple([int(diff[dim]/2)]*2)
                    else:
                        padding[dim] = tuple([int(np.floor(diff[dim]/2) + 1) , int(np.floor(diff[dim]/2))])
                
                return tuple(padding)

            params.Modelparam.FinalDimension = findFinalDimension(params)

            for ind, subj in enumerate(params.Input.Train.Subjects):
                params.Input.Train.Subjects[ind].Padding = findPadding1Subject(subj)

            for ind, subj in enumerate(params.Input.Test.Subjects):
                params.Input.Test.Subjects[ind].Padding = findPadding1Subject(subj)
                
            return params

        def searchSubejcts(params):
            
            def searchInd(subName, subject):
                origshape = nib.load(subject.Address + '/' + subName + '/WMnMPRAGE_bias_corr.nii.gz').shape
                shape = [int(a/params.Modelparam.downsampleFactor) for a in origshape]
                address = subject.Address + '/' + subName
                return subject(Name = subName, Padding = '', Address = address, Shape = shape, origShape = origshape)

            TrainList = [s for s in os.listdir(params.Input.Train.Address) if 'vimp' in s]
            TestList  = [s for s in os.listdir(params.Input.Test.Address) if 'vimp' in s]

            params.Input.Train.Subjects = [ searchInd(sub, params.Input.Train) for sub in TrainList ]
            params.Input.Test.Subjects = [ searchInd(sub, params.Input.Test) for sub in TestList ]  

            return params

        params = searchSubejcts(params)
        params = findPadding(params)

        return params

    def ReadingInput(Subjects, mode):

        def prepareInputImage(subj, image):
            im = zoom(imF.get_data(),0.5)
            im = np.pad(im, subj.Padding[:3], 'constant')[np.newaxis,...,np.newaxis]


        testData = {}
        images, masks = '', ''
        for ind, subj in tqdm(enumerate(Subjects),desc=mode):

            imF = nib.load(subj.Address + '/WMnMPRAGE_bias_corr.nii.gz')
            im = np.pad(imF.get_data(), subj.Padding[:3], 'constant')[np.newaxis,...,np.newaxis]

            msk = nib.load(subj.Address + '/temp/CropMask.nii.gz').get_data()
            msk = np.pad(msk, subj.Padding[:3], 'constant')[np.newaxis,...,np.newaxis]
            msk = np.concatenate((msk, 1-msk),axis=4).astype('float32')

            if np.min(subj.Padding) >= 0:
                    
                if 'train' in mode:
                    images = im      if ind == 0 else np.concatenate((images ,im      ),axis=0)
                    masks  = msk>0.5 if ind == 0 else np.concatenate((masks  ,msk>0.5 ),axis=0)
                
                testData[subj.Name] = DatatrainTest(Image=im, Mask=msk , Affine=imF.get_affine(), Header=imF.get_header())
            else:
                print('ERROR subject ',subj, ' dimensions exceed input dimension')

        class trainData:
            Images = images
            Masks = masks

        return testData, trainData
        
    params = preAnalysis(params)

    class TestData:
        Train = ''
        Test = ''

    class Data:
        testing = TestData()
        training = ''

    Data.testing.Train, Data.training = ReadingInput(params.Input.Train.Subjects, 'train')
    Data.testing.Test, _  = ReadingInput(params.Input.Test.Subjects, 'test')


        

    return Data, params



main()