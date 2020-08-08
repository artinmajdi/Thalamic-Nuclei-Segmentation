import numpy as np
# from imageio import imread
from random import shuffle
from tqdm import tqdm # , trange
import nibabel as nib
import shutil
import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import preprocess.normalizeA as normalizeA
from preprocess import applyPreprocess
# import matplotlib.pyplot as plt
from scipy import ndimage
# from shutil import copyfile
import h5py
from skimage.transform import AffineTransform , warp
# import pickle
# import skimage

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
            self.Mask  = Mask

    class testCase:
        def __init__(self, Image, Mask, OrigMask, Affine, Header, original_Shape):
            self.Image = Image
            self.Mask = Mask
            self.OrigMask  = OrigMask
            self.Affine = Affine
            self.Header = Header
            self.original_Shape = original_Shape

    return ImageLabel, data, trainCase, testCase

ImageLabel, data, trainCase, testCase = ClassesFunc()

def DatasetsInfo(DatasetIx):
    switcher = {
        1: ('SRI_3T', '/array/ssd/msmajdi/data/preProcessed/3T/SRI_3T'),
        2: ('SRI_ReSliced', '/array/ssd/msmajdi/data/preProcessed/3T/SRI_ReSliced'),
        3: ('croppingData', '/array/ssd/msmajdi/data/preProcessed/croppingData'),
        4: ('All_7T', '/array/ssd/msmajdi/data/preProcessed/7T/All_DBD'),
        5: ('20priors', '/array/ssd/msmajdi/data/preProcessed/7T/20priors'),
    }
    return switcher.get(DatasetIx, 'WARNING: Invalid dataset index')

def paddingNegativeFix(sz, Padding):
    padding = np.array([list(x) for x in Padding])
    crd = -1*padding
    padding[padding < 0] = 0
    crd[crd < 0] = 0
    Padding = tuple([tuple(x) for x in padding])

    # sz = im.shape
    # crd = crd[:3,:]
    for ix in range(3): crd[ix,1] = sz[ix] if crd[ix,1] == 0 else -crd[ix,1]

    return padding, crd

def loadDataset(params):

    def inputPreparationForUnet(im,subject2, params):

        def CroppingInput(im, Padding2):
            if np.min(Padding2) < 0:
                Padding2, crd = paddingNegativeFix(im.shape, Padding2)
                im = im[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]

            return np.pad(im, Padding2[:3], 'constant')

        # TODO I'm moving this to the stage before loading the image to increase the speed

        # if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and 1 not in params.WhichExperiment.Nucleus.Index:
        #     BB = subject2.NewCropInfo.OriginalBoundingBox
        #     im = im[BB[0][0]:BB[0][1]  ,  BB[1][0]:BB[1][1]  ,  BB[2][0]:BB[2][1]]
                        
        im = CroppingInput(im, subject2.Padding)
        # im = np.transpose(im, params.WhichExperiment.Dataset.slicingInfo.slicingOrder)
        im = np.transpose(im,[2,0,1])
        im = np.expand_dims(im ,axis=3).astype('float32')

        return im


    def read_cropped_inputs(params, subject, dirrectory):
        inputF = nib.load(dirrectory)
        if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and 1 not in params.WhichExperiment.Nucleus.Index:
            
            BB = subject.NewCropInfo.OriginalBoundingBox[params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse]
            input = inputF.dataobj[BB[0][0]:BB[0][1]  ,  BB[1][0]:BB[1][1]  ,  BB[2][0]:BB[2][1]]
        else:
            input = inputF.get_data()

        return inputF, input


    def readingImage(params, subject2, mode):

        def readingWithTranpose(Dirr , params):
            # ImageF = nib.load( Dirr)            
            ImageF, Image = read_cropped_inputs(params, subject2, Dirr)

            # if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and 1 not in params.WhichExperiment.Nucleus.Index:
            #     BB = subject2.NewCropInfo.OriginalBoundingBox[params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse]
            #     Image = ImageF.dataobj[BB[0][0]:BB[0][1]  ,  BB[1][0]:BB[1][1]  ,  BB[2][0]:BB[2][1]]
            # else:
            #     Image = ImageF.get_data()

            return ImageF, np.transpose(Image, params.WhichExperiment.Dataset.slicingInfo.slicingOrder)


        imF, im = readingWithTranpose(subject2.address + '/' + subject2.ImageProcessed + '.nii.gz' , params)


        im = inputPreparationForUnet(im, subject2, params)

        if params.preprocess.Normalize.per_Subject: im = normalizeA.main_normalize(params.preprocess.Normalize , im)

        # im = 1 - im
        return im, imF


    def readingNuclei(params, subject, imFshape):

        def backgroundDetector(masks):
            a = np.sum(masks,axis=3)
            background = np.zeros(masks.shape[:3])
            background[np.where(a == 0)] = 1
            background = np.expand_dims(background,axis=3)
            return background

        def readingOriginalMask(NucInd):
            nameNuclei, _,_ = smallFuncs.NucleiSelection(NucInd)
            inputMsk = subject.Label.address + '/' + nameNuclei + '_PProcessed.nii.gz'

            if os.path.exists(inputMsk):
                _, mask = read_cropped_inputs(params, subject, inputMsk)
                mask = smallFuncs.fixMaskMinMax(mask,nameNuclei)
            else:
                mask = np.zeros(10,10,10)
        
            return np.expand_dims(mask ,axis=3)

        for cnt, NucInd in enumerate(params.WhichExperiment.Nucleus.Index):

            origMsk1N = readingOriginalMask(NucInd)


            msk1N = np.transpose( np.squeeze(origMsk1N) , params.WhichExperiment.Dataset.slicingInfo.slicingOrder)
            msk1N = inputPreparationForUnet(msk1N, subject, params)

            origMsk = origMsk1N if cnt == 0 else np.concatenate((origMsk, origMsk1N) ,axis=3).astype('float32')
            msk = msk1N if cnt == 0 else np.concatenate((msk,msk1N),axis=3).astype('float32')

        if params.WhichExperiment.HardParams.Model.Method.havingBackGround_AsExtraDimension:
            background = backgroundDetector(msk)
            msk = np.concatenate((msk, background),axis=3).astype('float32')

        return origMsk , msk

    def Error_MisMatch_In_Dim_ImageMask(subject, mode, nameSubject):
        AA = subject.address.split(os.path.dirname(subject.address) + '/')
        shutil.move(subject.address, os.path.dirname(subject.address) + '/' + 'ERROR_' + AA[1])
        print('WARNING:', mode , nameSubject, ' image and mask have different shape sizes')
    
    def saveHDf5(g1 , im , msk , origMsk , imF , subject):
        g = g1.create_group(subject.subjectName)
        g.create_dataset(name='Image'    ,data=im)
        g.create_dataset(name='Mask'     ,data=msk )
        g.create_dataset(name='OrigMask' ,data=(origMsk).astype('float32') )
        g.attrs['original_Shape'] = imF.shape
        g.attrs['Affine'] = imF.get_affine()
        g.attrs['address'] = subject.address   
   
    def main_ReadingDataset(params):

        def sagittalFlag():
            return params.WhichExperiment.Nucleus.Index[0] == 1 and params.WhichExperiment.Dataset.slicingInfo.slicingDim == 2
        
        def trainFlag():
            Flag_TestOnly = params.preprocess.TestOnly
            Flag_TrainDice = params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data
            Flag_cascadeMethod = 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and int(params.WhichExperiment.Nucleus.Index[0]) == 1
            Flag_notEmpty = params.directories.Train.Input.Subjects
            measure_train = params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data

            # flag_testOnly = False if whichExperiment.TestOnly and (modeData == 'train') else True
            # return (  (not Flag_TestOnly) or Flag_TrainDice ) and Flag_notEmpty
            return (not Flag_TestOnly or measure_train) and Flag_notEmpty
        
        Th = 0.5*params.WhichExperiment.HardParams.Model.LabelMaxValue

        def save_hdf5_subject_List(h , Tag , List):
            List = [n.encode("ascii", "ignore") for n in List]
            h.create_dataset(Tag ,(len(List),1) , 'S10', List)  
            
        def separatingConcatenatingIndexes(Data, sjList, mode):

            Sz0 = 0
            for nameSubject in sjList: Sz0 += Data[nameSubject].Image.shape[0]
            images = np.zeros(  (  tuple([Sz0]) + Data[list(Data)[0]].Image.shape[1:] )  )
            masks = np.zeros(  (  tuple([Sz0]) + Data[list(Data)[0]].Mask.shape[1:] )  )

            d1 = 0
            for _, nameSubject in enumerate(tqdm(sjList,desc='concatenating: ' + mode)):
                im, msk = Data[nameSubject].Image  , Data[nameSubject].Mask
                
                images[d1:d1+im.shape[0] ,...] = im
                masks[d1:d1+im.shape[0] ,...]  = msk
                d1 += im.shape[0]
            
            return trainCase(Image=images, Mask=masks.astype('float32'))
            
        def separateTrainVal_and_concatenateTrain(DataAll):   

            def func_normalize(DataAll):
                if DataAll.Train:      DataAll.Train.Image = normalizeA.main_normalize(params.preprocess.Normalize , DataAll.Train.Image)
                if DataAll.Validation: DataAll.Validation.Image = normalizeA.main_normalize(params.preprocess.Normalize , DataAll.Validation.Image)
                
                for nameSubject in list(DataAll.Train_ForTest):
                    DataAll.Train_ForTest[nameSubject].Image = normalizeA.main_normalize(params.preprocess.Normalize , DataAll.Train_ForTest[nameSubject].Image)

                for nameSubject in list(DataAll.Test):
                    DataAll.Test[nameSubject].Image = normalizeA.main_normalize(params.preprocess.Normalize , DataAll.Test[nameSubject].Image)
                
                return DataAll

            TrainList, ValList = percentageDivide(params.WhichExperiment.Dataset.Validation.percentage, list(params.directories.Train.Input.Subjects), params.WhichExperiment.Dataset.randomFlag)

            save_hdf5_subject_List(params.h5 , 'trainList' , TrainList )

            if params.WhichExperiment.Dataset.Validation.fromKeras or params.WhichExperiment.HardParams.Model.Method.Use_TestCases_For_Validation:
                DataAll.Train = separatingConcatenatingIndexes(DataAll.Train_ForTest, list(DataAll.Train_ForTest),'train')                
                DataAll.Validation = ''            
            else:                
                DataAll.Train = separatingConcatenatingIndexes(DataAll.Train_ForTest, TrainList,'train')
                DataAll.Validation = separatingConcatenatingIndexes(DataAll.Train_ForTest, ValList,'validation')
                
                save_hdf5_subject_List(params.h5 , 'valList' , ValList )

            if params.preprocess.Normalize.per_Dataset: DataAll = func_normalize(DataAll)

            return DataAll

        def readingAllSubjects(Subjects, mode):

            def ErrorInPaddingCheck(subject):
                ErrorFlag = False
                if np.min(subject.Padding) < 0:
                    if np.min(subject.Padding) < -params.WhichExperiment.HardParams.Model.paddingErrorPatience:
                        AA = subject.address.split(os.path.dirname(subject.address) + '/')
                        shutil.move(subject.address, os.path.dirname(subject.address) + '/' + 'ERROR_' + AA[1])
                        print('WARNING: subject: ',subject.subjectName,' size is out of the training network input dimensions')
                        ErrorFlag = True
                    else:
                        Dirsave = smallFuncs.mkDir(params.directories.Test.Result + '/' + subject.subjectName,)
                        np.savetxt(Dirsave + '/paddingError.txt', subject.Padding, fmt='%d')
                        print('WARNING: subject: ',subject.subjectName,' padding error patience activated, Error:', np.min(subject.Padding))
                return ErrorFlag

            def upsample_Image(Image, Mask , scale):
                szI = Image.shape
                szM = Mask.shape
                
                Image3 = np.zeros( (szI[0] , scale*szI[1] , scale*szI[2] , szI[3])  )
                Mask3  = np.zeros( (szM[0] , scale*szM[1] , scale*szM[2] , szM[3])  )

                newShape = (scale*szI[1] , scale*szI[2])

                # for i in range(Image.shape[2]):
                #     Image2[...,i] = scipy.misc.imresize(Image[...,i] ,size=newShape[:2] , interp='cubic')
                #     Mask2[...,i]  = scipy.misc.imresize( (Mask[...,i] > 0.5).astype(np.float32) ,size=newShape[:2] , interp='bilinear')

                tform = AffineTransform(scale=(scale, scale))
                for i in range(Image.shape[0]):

                    for ch in range(Image3.shape[3]):
                        Image3[i ,: ,: ,ch] = warp( np.squeeze(Image[i ,: ,: ,ch]), tform.inverse, output_shape=newShape, order=3)

                    for ch in range(Mask3.shape[3]):
                        Mask3[i ,: ,: ,ch]  = warp( (np.squeeze(Mask[i ,: ,: ,ch]) > 0.5).astype(np.float32) ,  tform.inverse, output_shape=newShape, order=0)
                
                return Image3 , Mask3
                
            Data = {}
            # g1 = params.h5.create_group(mode)
            for nameSubject, subject in tqdm(Subjects.items(), desc='Loading ' + mode):

                if ErrorInPaddingCheck(subject): continue

                im, imF = readingImage(params, subject, mode)
                origMsk , msk = readingNuclei(params, subject, imF.shape)

                msk = msk>Th
                origMsk = origMsk>Th

                if params.WhichExperiment.HardParams.Model.Upsample.Mode:
                    scale = params.WhichExperiment.HardParams.Model.Upsample.Scale
                    im, msk = upsample_Image(im, msk , scale)    


                if im[...,0].shape == msk[...,0].shape:
                    Data[nameSubject] = testCase(Image=im, Mask=msk ,OrigMask=(origMsk).astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)
                    # saveHDf5(g1 , im , msk , origMsk , imF , subject)
                                        
                else: 
                    Error_MisMatch_In_Dim_ImageMask(subject , mode, nameSubject)


            return Data

        def readValidation(DataAll): 
            if params.WhichExperiment.HardParams.Model.Method.Use_TestCases_For_Validation:
                Read = params.WhichExperiment.Dataset.ReadTrain
                Val_Indexes = list(DataAll.Test)
            
                if Read.CSFn2 and Read.CSFn1:              Val_Indexes = [s for s in Val_Indexes if ('CSFn1' not in s)]
                # elif (Read.Main or Read.ET) and Read.SRI:  Val_Indexes = [s for s in Val_Indexes if ('SRI' not in s)]

                DataAll.Validation = separatingConcatenatingIndexes(DataAll.Test, Val_Indexes, 'validation')                
                save_hdf5_subject_List(params.h5 , 'valList' , Val_Indexes)
            return DataAll
            
        DataAll = data()
        
        if trainFlag():
            DataAll.Train_ForTest = readingAllSubjects(params.directories.Train.Input.Subjects, 'train')
            DataAll = separateTrainVal_and_concatenateTrain( DataAll )


        if params.directories.Test.Input.Subjects: 
            DataAll.Test = readingAllSubjects(params.directories.Test.Input.Subjects, 'test')
            # save_hdf5_subject_List(params.h5 , 'testList' , list(DataAll.Test))

            DataAll = readValidation(DataAll)

        if sagittalFlag():
            if trainFlag(): DataAll.Sagittal_Train_ForTest = readingAllSubjects(params.directories.Train.Input_Sagittal.Subjects, 'trainS')
            DataAll.Sagittal_Test          = readingAllSubjects(params.directories.Test.Input_Sagittal.Subjects , 'testS' )               
                
        return DataAll

    params = preAnalysis(params)
    
    smallFuncs.mkDir(params.directories.Test.Result)
    params.h5 = h5py.File(params.directories.Test.Result + '/Data.hdf5','w')
    Data = main_ReadingDataset(params)
    params.h5.close()
    # saveHDf5(Data)


    return Data, params

def percentageDivide(percentage, subjectsList, randomFlag):

    L = len(subjectsList)
    indexes = np.array(range(L))

    if randomFlag: shuffle(indexes)
    per = int( percentage * L )
    if per == 0 and L > 1: per = 1

    TestVal_List = [subjectsList[i] for i in indexes[:per]]
    Train_List = [subjectsList[i] for i in indexes[per:]]

    return Train_List, TestVal_List

def movingFromDatasetToExperiments(params):

    if len(os.listdir(params.directories.Train.address)) != 0 or len(os.listdir(params.directories.Test.address)) != 0:
        print('*** DATASET ALREADY EXIST; PLEASE REMOVE \'train\' & \'test\' SUBFOLDERS ***')
        sys.exit

    else:
        List = smallFuncs.listSubFolders(params.WhichExperiment.Dataset.address, params)

        TestParams  = params.WhichExperiment.Dataset.Test
        _, TestList = percentageDivide(TestParams.percentage, List, params.WhichExperiment.Dataset.randomFlag) if 'percentage' in TestParams.mode else TestParams.subjects
        for subject in List:

            DirOut, _ = (params.directories.Test.address , 'test') if subject in TestList else (params.directories.Train.address, 'train')

            if not os.path.exists(DirOut + '/' + subject):
                shutil.copytree(params.WhichExperiment.Dataset.address + '/' + subject  ,  DirOut + '/' + subject)

    return True

def preAnalysis(params):

    def find_AllInputSizes(params):

        def newCropedSize(subject, params, mode):
            
            # '_sd' + str(UserInfo['simulation'].slicingDim[0])
            def readingCascadeCropSizes(subject):

                dirr = params.directories.Test.Result                
                if 'train' in mode: dirr += '/TrainData_Output'
                
                BBf = np.loadtxt(dirr + '/' + subject.subjectName  + '/BB_' + params.WhichExperiment.HardParams.Model.Method.ReferenceMask + '.txt',dtype=int)
                BB = BBf[:,:2]
                BBd = BBf[:,2:]

                #! because on the slicing direction we don't want the extra dilated effect to be considered
                BBd[params.WhichExperiment.Dataset.slicingInfo.slicingDim] = BB[params.WhichExperiment.Dataset.slicingInfo.slicingDim]
                BBd = BBd[params.WhichExperiment.Dataset.slicingInfo.slicingOrder]
                return BBd

            if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and 1 not in params.WhichExperiment.Nucleus.Index:
                BB = readingCascadeCropSizes(subject)

                origSize = np.array( nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz').shape )
                origSize = origSize[params.WhichExperiment.Dataset.slicingInfo.slicingOrder]


                subject.NewCropInfo.OriginalBoundingBox = BB
                subject.NewCropInfo.PadSizeBackToOrig   = tuple([ tuple([BB[d][0] , origSize[d]-BB[d][1]]) for d in range(3) ])

                Shape = np.array([BB[d][1]-BB[d][0] for d  in range(3)])
            else:
                Shape = np.array( nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz').shape )
                Shape = Shape[params.WhichExperiment.Dataset.slicingInfo.slicingOrder]

            # Shape = tuple(Shape[params.WhichExperiment.Dataset.slicingInfo.slicingOrder])
            return Shape, subject

        def loopOverAllSubjects(Input, mode):
            inputSize = []
            for sj in Input.Subjects:
                Shape, Input.Subjects[sj] = newCropedSize(Input.Subjects[sj], params, mode)
                inputSize.append(Shape)

            Input.inputSizes = np.array(inputSize)
            return Input

        if params.directories.Train.Input.Subjects: params.directories.Train.Input = loopOverAllSubjects(params.directories.Train.Input, 'train')
        if params.directories.Test.Input.Subjects: params.directories.Test.Input  = loopOverAllSubjects(params.directories.Test.Input, 'test')

        if params.WhichExperiment.Nucleus.Index[0] == 1 and params.WhichExperiment.Dataset.slicingInfo.slicingDim == 2:
            if params.directories.Train.Input_Sagittal.Subjects: params.directories.Train.Input_Sagittal = loopOverAllSubjects(params.directories.Train.Input_Sagittal, 'train')
            if params.directories.Test.Input_Sagittal.Subjects:  params.directories.Test.Input_Sagittal  = loopOverAllSubjects(params.directories.Test.Input_Sagittal, 'test')
                            

        return params

    def find_correctNumLayers(params):

        HardParams = params.WhichExperiment.HardParams
        
        if params.WhichExperiment.HardParams.Model.architectureType != 'FCN':    # 'Cascade' in HardParams.Model.Method.Type and 

            def func_MinInputSize(params):
                if params.WhichExperiment.Dataset.InputPadding.Automatic: 
                    inputSizes = params.directories.Test.Input.inputSizes if params.WhichExperiment.TestOnly else np.concatenate((params.directories.Train.Input.inputSizes , params.directories.Test.Input.inputSizes),axis=0)

                    return np.min(inputSizes, axis=0)
                else:
                    return params.WhichExperiment.Dataset.InputPadding.HardDimensions

            MinInputSize = func_MinInputSize(params)

            kernel_size = HardParams.Model.Layer_Params.ConvLayer.Kernel_size.conv
            num_Layers  = HardParams.Model.num_Layers

            if params.WhichExperiment.HardParams.Model.Upsample.Mode:
                MinInputSize = MinInputSize*params.WhichExperiment.HardParams.Model.Upsample.Scale                 


            params.WhichExperiment.HardParams.Model.num_Layers_changed = False
            dim = HardParams.Model.Method.InputImage2Dvs3D
            
            # ! check if the figure map size at the most bottom layer is bigger than convolution kernel size                
            if np.min(MinInputSize[:dim] - np.multiply( kernel_size,(2**(num_Layers - 1)))) < 0: 
                params.WhichExperiment.HardParams.Model.num_Layers = int(np.floor( np.log2(np.min( np.divide(MinInputSize[:dim],kernel_size) )) + 1))
                print('WARNING: INPUT IMAGE SIZE IS TOO SMALL FOR THE NUMBER OF LAYERS')
                print('# LAYERS  OLD:',num_Layers  ,  ' =>  NEW:',params.WhichExperiment.HardParams.Model.num_Layers)
                params.WhichExperiment.HardParams.Model.num_Layers_changed = True
            
        return params

    def find_PaddingValues(params):

        def findingPaddedInputSize(params):
            inputSizes = params.directories.Test.Input.inputSizes if params.WhichExperiment.TestOnly else np.concatenate((params.directories.Train.Input.inputSizes , params.directories.Test.Input.inputSizes),axis=0)
            # inputSizes = np.concatenate((params.directories.Train.Input.inputSizes , params.directories.Test.Input.inputSizes),axis=0)  
            
            num_Layers = params.WhichExperiment.HardParams.Model.num_Layers
            L = num_Layers if 'SegNet' in params.WhichExperiment.HardParams.Model.architectureType else num_Layers - 1
                
            a = 2**(L) 
            return [  int(a * np.ceil(s / a)) if s % a != 0 else s for s in np.max(inputSizes, axis=0) ]
            
        def findingSubjectsFinalPaddingAmount(wFolder, Input, params):

            def applyingPaddingDimOnSubjects(params, Input):
                fullpadding = params.WhichExperiment.HardParams.Model.InputDimensions - Input.inputSizes
                md = np.mod(fullpadding,2)
                for sn, name in enumerate(list(Input.Subjects)):
                    padding = [tuple([0,0])]*4
                    
                    for dim in range(params.WhichExperiment.HardParams.Model.Method.InputImage2Dvs3D): # params.WhichExperiment.Dataset.slicingInfo.slicingOrder[:2]:
                        if md[sn, dim] == 0:
                            padding[dim] = tuple([int(fullpadding[sn,dim]/2)]*2)
                        else:
                            padding[dim] = tuple([int(np.floor(fullpadding[sn,dim]/2) + 1) , int(np.floor(fullpadding[sn,dim]/2))])

                    if np.min(tuple(padding)) < 0:
                        print('---')
                    Input.Subjects[name].Padding = tuple(padding)

                return Input

            return applyingPaddingDimOnSubjects(params, Input)

        AA = findingPaddedInputSize( params ) if params.WhichExperiment.Dataset.InputPadding.Automatic else params.WhichExperiment.Dataset.InputPadding.HardDimensions
        params.WhichExperiment.HardParams.Model.InputDimensions = AA

        if params.directories.Train.Input.Subjects: params.directories.Train.Input = findingSubjectsFinalPaddingAmount('Train', params.directories.Train.Input, params)
        if params.directories.Test.Input.Subjects:  params.directories.Test.Input  = findingSubjectsFinalPaddingAmount('Test', params.directories.Test.Input, params)

        if params.WhichExperiment.Nucleus.Index[0] == 1 and params.WhichExperiment.Dataset.slicingInfo.slicingDim == 2:
            if params.directories.Train.Input_Sagittal.Subjects: params.directories.Train.Input_Sagittal = findingSubjectsFinalPaddingAmount('Train', params.directories.Train.Input_Sagittal, params)
            if params.directories.Test.Input_Sagittal.Subjects:  params.directories.Test.Input_Sagittal  = findingSubjectsFinalPaddingAmount('Test', params.directories.Test.Input_Sagittal, params)
                            
        return params

    params = find_AllInputSizes(params)
    params = find_correctNumLayers(params)
    params = find_PaddingValues(params)

    # saveUserParams(params)

    return params
