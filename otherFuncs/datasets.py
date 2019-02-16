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
from scipy import ndimage
from shutil import copyfile

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
        Data, params = readingFromExperiments(params)

    _, Data.Info.Height, Data.Info.Width, _ = Data.Test[list(Data.Test)[0]].Image.shape if params.preprocess.TestOnly else Data.Train.Image.shape
    params.WhichExperiment.HardParams.Model.imageInfo = Data.Info
    return Data, params

def fashionMnist(params):

    def one_hot(a, num_classes):
        return np.eye(num_classes)[a]

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

# TODO: add the saving images with the format mahesh said
# TODO: maybe add the ability to crop the test cases with bigger sizes than network input dimention accuired from train datas
def readingFromExperiments(params):
          
    def inputPreparationForUnet(im,subject, params):

        def CroppingInput(im, subjectB):
            
            if np.min(subjectB.Padding) < 0:    
                padding = np.array([list(x) for x in subjectB.Padding])                
                crd = -1*padding 
                padding[padding < 0] = 0
                crd[crd < 0] = 0
                subjectB.Padding = tuple([tuple(x) for x in padding])

                sz = im.shape
                crd = crd[:len(sz),:]
                for ix in range(len(sz)): crd[ix,1] = sz[ix] if crd[ix,1] == 0 else -crd[ix,1]
                        
                im = im[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]

                
            im = np.pad(im, subjectB.Padding[:3], 'constant')

            return im

        if 'cascadeThalamus' in params.WhichExperiment.HardParams.Model.Idea and 1 not in params.WhichExperiment.Nucleus.Index: 
            # subject.NewCropInfo.PadSizeBackToOrig
            BB = subject.NewCropInfo.OriginalBoundingBox            
            im = im[BB[0][0]:BB[0][1]  ,  BB[1][0]:BB[1][1]  ,  BB[2][0]:BB[2][1]] 

        aa = subject.address.split('train/') if len(subject.address.split('train/')) == 2 else subject.address.split('test/')

        im = np.transpose(im, params.WhichExperiment.Dataset.slicingInfo.slicingOrder)
        im = CroppingInput(im, subject)    
        
        im = np.transpose(im,[2,0,1])
        im = np.expand_dims(im ,axis=3).astype('float32')
        
        return im
        
    def readingImage(params, subject):
        imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')
        im = inputPreparationForUnet(imF.get_data(), subject, params)
        im = normalizeA.main_normalize(params.preprocess.Normalize , im)
        return im, imF

    def readingNuclei(params, subject, imFshape):

        def backgroundDetector(masks):
            a = np.sum(masks,axis=3)
            background = np.zeros(masks.shape[:3])
            background[np.where(a == 0)] = 1
            background = np.expand_dims(background,axis=3)
            return background

        for cnt, NucInd in enumerate(params.WhichExperiment.Nucleus.Index):
            nameNuclei, _,_ = smallFuncs.NucleiSelection(NucInd)
            inputMsk = subject.Label.address + '/' + nameNuclei + '_PProcessed.nii.gz'

            origMsk1N = nib.load(inputMsk).get_data() if os.path.exists(inputMsk) else np.zeros(imFshape) 
            msk1N = inputPreparationForUnet(origMsk1N, subject, params)
            origMsk1N = np.expand_dims(origMsk1N ,axis=3)

            origMsk = origMsk1N if cnt == 0 else np.concatenate((origMsk, origMsk1N) ,axis=3).astype('float32')
            msk = msk1N if cnt == 0 else np.concatenate((msk,msk1N),axis=3).astype('float32')

        background = backgroundDetector(msk)
        msk = np.concatenate((msk, background),axis=3).astype('float32')
        
        return origMsk , msk
            
    def Error_MisMatch_In_Dim_ImageMask(cntSkipped,subject, mode, nameSubject):
        AA = subject.address.split(os.path.dirname(subject.address) + '/')
        shutil.move(subject.address, os.path.dirname(subject.address) + '/' + 'ERROR_' + AA[1])
        print('WARNING:', mode , cntSkipped + 1 , nameSubject, ' image and mask have different shape sizes')
        return cntSkipped + 1

    def preAnalysis(params):

        def find_PaddingValues(params):

            def findingSubjectsFinalPaddingAmount(wFolder, Input, params):

                def findingPaddedInputSize(inputSize, params):
                    MaxInputSize = np.max(inputSize, axis=0)
                    new_inputSize = MaxInputSize

                    a = 2**(params.WhichExperiment.HardParams.Model.num_Layers - 1)
                    for dim in range(2):
                        # checking how much we need to pad the input image to make sure the we don't lose any information because of odd dimension sizes
                        if MaxInputSize[dim] % a != 0:
                            new_inputSize[dim] = a * np.ceil(MaxInputSize[dim] / a)

                    return new_inputSize
                
                def applyingPaddingDimOnSubjects(params, inputSize, Subjects):
                    fullpadding = params.WhichExperiment.HardParams.Model.InputDimensions[:2] - inputSize[:,:2]
                    md = np.mod(fullpadding,2)

                    for sn, name in enumerate(list(Subjects)):
                        padding = [np.zeros(2)]*4
                        for dim in range(2):
                            if md[sn, dim] == 0:
                                padding[dim] = tuple([int(fullpadding[sn,dim]/2)]*2)
                            else:
                                padding[dim] = tuple([int(np.floor(fullpadding[sn,dim]/2) + 1) , int(np.floor(fullpadding[sn,dim]/2))])

                        padding[2] = tuple([0,0])
                        padding[3] = tuple([0,0])
                        Subjects[name].Padding = tuple(padding)

                    return Subjects

                if 'Train' in wFolder: params.WhichExperiment.HardParams.Model.InputDimensions = findingPaddedInputSize( Input.inputSizes, params )

                # else:
                #     new_inputSize = params.WhichExperiment.HardParams.Model.InputDimensions
                
                #! finding the amount of padding for each subject in each direction
                Input.Subjects = applyingPaddingDimOnSubjects(params, Input.inputSizes, Input.Subjects)

                return Input
                
            params.directories.Train.Input = findingSubjectsFinalPaddingAmount('Train', params.directories.Train.Input, params)
            params.directories.Test.Input  = findingSubjectsFinalPaddingAmount('Test', params.directories.Test.Input, params)

            return params

        def find_AllInputSizes(params):

            def newCropedSize(subject, params):

                def readingThalamicCropSizes(subject , slicingDirection):
                    BB = np.loadtxt(subject.Temp.address + '/BB.txt',dtype=int)
                    BBd = np.loadtxt(subject.Temp.address + '/BBd.txt',dtype=int)

                    #! ebcause on the slicing direction we don't want the extra dilated effect to be considered
                    BBd[slicingDirection] = BB[slicingDirection]
                                    
                    return BBd
                    
                if 'cascadeThalamus' in params.WhichExperiment.HardParams.Model.Idea and 1 not in params.WhichExperiment.Nucleus.Index: 
                    BB = readingThalamicCropSizes(subject, params.WhichExperiment.Dataset.slicingInfo.slicingDim)

                    origSize = np.array( nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz').shape )

                    subject.NewCropInfo.OriginalBoundingBox = BB            
                    subject.NewCropInfo.PadSizeBackToOrig   = tuple([ tuple([BB[d][0] , origSize[d]-BB[d][1]]) for d in range(3) ])

                    Shape = np.array([BB[d][1]-BB[d][0] for d  in range(3)])     
                else:
                    Shape = np.array( nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz').shape )

                Shape = tuple(Shape[params.WhichExperiment.Dataset.slicingInfo.slicingOrder])
                return Shape, subject
            def loopOverAllSubjects(Input):
                inputSize = []
                for sj in Input.Subjects:
                    # if sj == 'vimp2_0699_04302014':
                    #     print('----')

                    Shape, Input.Subjects[sj] = newCropedSize( Input.Subjects[sj], params)
                    inputSize.append(Shape)

                Input.inputSizes = np.array(inputSize)
                return Input

            params.directories.Train.Input = loopOverAllSubjects(params.directories.Train.Input)
            params.directories.Test.Input  = loopOverAllSubjects(params.directories.Test.Input)
                                                
            return params
                    
        def find_correctNumLayers(params):

            HardParams = params.WhichExperiment.HardParams

            MinInputSize = np.min(params.directories.Train.Input.inputSizes, axis=0)
            kernel_size = HardParams.Model.ConvLayer.Kernel_size.conv
            num_Layers  = HardParams.Model.num_Layers

            if np.min(MinInputSize[:2] - np.multiply( kernel_size,(2**(num_Layers - 1)))) < 0:  # ! check if the figure map size at the most bottom layer is bigger than convolution kernel size
                print('WARNING: INPUT IMAGE SIZE IS TOO SMALL FOR THE NUMBER OF LAYERS')
                num_Layers = int(np.floor( np.log2(np.min( np.divide(MinInputSize[:2],kernel_size) )) + 1))
                print('# LAYERS  OLD:',HardParams.Model.num_Layers  ,  ' =>  NEW:',num_Layers)

            params.WhichExperiment.HardParams.Model.num_Layers = num_Layers
            return params

        params = find_AllInputSizes(params)
        params = find_correctNumLayers(params)
        params = find_PaddingValues(params)    

        return params

    def ErrorInPaddingCheck(params, subject, nameSubject):
        ErrorFlag = False
        if np.min(subject.Padding) < 0:
            if np.min(subject.Padding) < -params.WhichExperiment.HardParams.Model.paddingErrorPatience:
                AA = subject.address.split(os.path.dirname(subject.address) + '/')
                shutil.move(subject.address, os.path.dirname(subject.address) + '/' + 'ERROR_' + AA[1])
                print('WARNING: subject: ',nameSubject,' size is out of the training network input dimensions')
                ErrorFlag = True
            else:
                Dirsave = smallFuncs.mkDir(params.directories.Test.Result + '/' + nameSubject)
                np.savetxt(Dirsave + '/paddingError.txt', subject.Padding, fmt='%d')
                print('WARNING: subject: ',nameSubject,' padding error patience activated, Error:', np.min(subject.Padding))
        return ErrorFlag

    def loopOver_ReadingInput(params):

        def saveTrainDataFinal(DataAll, TrainData, images, masks):
            DataAll.Train_ForTest = TrainData

            if params.WhichExperiment.Dataset.Validation.fromKeras:
                DataAll.Train = trainCase(Image=images, Mask=masks.astype('float32'))
            else:
                DataAll.Train, DataAll.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage, images, masks, params.WhichExperiment.Dataset.randomFlag)
            return DataAll

        def checkSimulationMode(params, mode):
            if 'train' in mode and params.preprocess.TestOnly and not params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data:
                return True
            else:
                return False

        DataAll = data()
        Th = 0.5*params.WhichExperiment.HardParams.Model.LabelMaxValue
        TestData, TrainData = {}, {}
        for mode in ['train','test']:
            
            if checkSimulationMode(params, mode): continue
            Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects
            cntSkipped, indTrain = 0, 0
            for _, nameSubject in tqdm(enumerate(Subjects), desc='Loading Dataset: ' + mode):
                # subject = Subjects[nameSubject]
                    
                if ErrorInPaddingCheck(params, Subjects[nameSubject], nameSubject): continue
                            
                im, imF = readingImage(params, Subjects[nameSubject])
                origMsk , msk = readingNuclei(params, Subjects[nameSubject], imF.shape)

                if im[...,0].shape == msk[...,0].shape:
                    if 'train' in mode:                        
                        images = im     if indTrain == 0 else np.concatenate((images,im    ),axis=0)
                        masks  = msk>Th if indTrain == 0 else np.concatenate((masks,msk>Th ),axis=0)
                        TrainData[nameSubject] = testCase(Image=im, Mask=msk ,OrigMask=origMsk.astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)
                        indTrain = indTrain + 1

                    elif 'test' in mode:
                        TestData[nameSubject]  = testCase(Image=im, Mask=msk ,OrigMask=origMsk.astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)
                else:
                    cntSkipped = Error_MisMatch_In_Dim_ImageMask(cntSkipped, Subjects[nameSubject] , mode, nameSubject)


            if 'train' in mode:
                TrainData = saveTrainDataFinal(DataAll, TrainData, images, masks)
                # DataAll.Train_ForTest = TrainData

                # if params.WhichExperiment.Dataset.Validation.fromKeras:
                #     DataAll.Train = trainCase(Image=images, Mask=masks.astype('float32'))
                # else:
                #     DataAll.Train, DataAll.Validation = TrainValSeperate(params.WhichExperiment.Dataset.Validation.percentage, images, masks, params.WhichExperiment.Dataset.randomFlag)
            else:
                DataAll.Test = TestData

        _, DataAll.Info.Height, DataAll.Info.Width, _ = DataAll.Test[list(DataAll.Test)[0]].Image.shape if params.preprocess.TestOnly else DataAll.Train.Image.shape
        params.WhichExperiment.HardParams.Model.imageInfo = DataAll.Info
        return DataAll, params

    params = preAnalysis(params)
    Data, params = loopOver_ReadingInput(params)

    return Data, params

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

        def listAugmentationFolders(mode, params):
            Dir_Aug1 = params.WhichExperiment.Dataset.address + '/Augments/' + mode
            flag_Aug = os.path.exists(Dir_Aug1)

            ListAugments = smallFuncs.listSubFolders(Dir_Aug1, params) if flag_Aug else list('')

            return flag_Aug, {'address': Dir_Aug1 , 'list': ListAugments , 'mode':mode}

        flagAg, AugDataL = np.zeros(3), list(np.zeros(3))
        if params.Augment.Mode:
            if params.Augment.Linear.Rotation.Mode:     flagAg[0], AugDataL[0] = listAugmentationFolders('Linear_Rotation', params)
            if params.Augment.Linear.Shift.Mode:        flagAg[1], AugDataL[1] = listAugmentationFolders('Linear_Shift', params)
            if params.Augment.NonLinear.Mode: flagAg[2], AugDataL[2] = listAugmentationFolders('NonLinear', params)

        return flagAg, AugDataL
        
    def copyAugmentData(DirOut, AugDataL, subject):
        if 'NonLinear' in AugDataL['mode']: AugDataL['list'] = [i for i in AugDataL['list'] if subject in i.split('Ref_')[0]]

        for subjectsAgm in AugDataL['list']:
            if subject in subjectsAgm: shutil.copytree(AugDataL['address'] + '/' + subjectsAgm  ,  DirOut + '/' + subjectsAgm)
                    
    if len(os.listdir(params.directories.Train.address)) != 0 or len(os.listdir(params.directories.Test.address)) != 0:
        print('*** DATASET ALREADY EXIST; PLEASE REMOVE \'train\' & \'test\' SUBFOLDERS ***')
        sys.exit
    
    else:
        List = smallFuncs.listSubFolders(params.WhichExperiment.Dataset.address, params)
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

        # params = smallFuncs.inputNamesCheck(params, 'experiment')
        
    return True

   



   

