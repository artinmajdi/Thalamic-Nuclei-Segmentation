import numpy as np
from imageio import imread
from random import shuffle
from tqdm import tqdm, trange
import nibabel as nib
import shutil
import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import preprocess.normalizeA as normalizeA
import preprocess.applyPreprocess as applyPreprocess
import matplotlib.pyplot as plt
from scipy import ndimage
from shutil import copyfile
import h5py
import pickle
import skimage

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

        if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and 1 not in params.WhichExperiment.Nucleus.Index:
            # subject.NewCropInfo.PadSizeBackToOrig
            BB = subject2.NewCropInfo.OriginalBoundingBox
            im = im[BB[0][0]:BB[0][1]  ,  BB[1][0]:BB[1][1]  ,  BB[2][0]:BB[2][1]]

            # def save_4Nuclei_CroppedMask():
            # im = nib.load(subject2.address + '/' + subject2.ImageProcessed + '.nii.gz').slicer[BB[0][0]:BB[0][1]    ,  BB[1][0]:BB[1][1]  ,  BB[2][0]:BB[2][1]]  
            # msk = nib.load(subject2.Label.address + '/' + subject2.Label.LabelProcessed.replace('ImClosed_','') + '_DifferentLabels.nii.gz').slicer[BB[0][0]:BB[0][1]    ,  BB[1][0]:BB[1][1]  ,  BB[2][0]:BB[2][1]]  
            # a = smallFuncs.Nuclei_Class(1,'HCascade').HCascade_Parents_Identifier([params.UserInfo['simulation'].nucleus_Index])
            # b = smallFuncs.Nuclei_Class(a[0],'HCascade').name
            # im = nib.load(subject2.address + '/' + subject2.ImageProcessed + '.nii.gz').slicer[BB[0][0]:BB[0][1]    ,  BB[1][0]:BB[1][1]  ,  BB[2][0]:BB[2][1]]  
            # msk = nib.load(subject2.Label.address + '/' + b.replace('_ImClosed','') + '_PProcessed_DifferentLabels.nii.gz').slicer[BB[0][0]:BB[0][1]    ,  BB[1][0]:BB[1][1]  ,  BB[2][0]:BB[2][1]]  

            # nib.save(im,'/array/ssd/msmajdi/experiments/'  + 'Image_' + subject2.Label.LabelProcessed.split('_ImClosed_PProcessed')[0] + '_Stage2.nii.gz' )
            # nib.save(msk,'/array/ssd/msmajdi/experiments/' + 'Mask_'  + subject2.Label.LabelProcessed.split('_ImClosed_PProcessed')[0] + '_DiffLabels_Stage2.nii.gz' )
            # print('----')
                        
        im = CroppingInput(im, subject2.Padding)
        # im = np.transpose(im, params.WhichExperiment.Dataset.slicingInfo.slicingOrder)
        im = np.transpose(im,[2,0,1])
        im = np.expand_dims(im ,axis=3).astype('float32')

        return im

    def readingImage(params, subject2, mode):

        def readingWithTranpose(Dirr , params):
            ImageF = nib.load( Dirr)
            Image = ImageF.get_data()
            return ImageF, np.transpose(Image, params.WhichExperiment.Dataset.slicingInfo.slicingOrder)

        def func_Multiply_By_Thalmaus(imm):

            def dilateMask(mask):
                struc = ndimage.generate_binary_structure(3,2)
                struc = ndimage.iterate_structure(struc, params.WhichExperiment.Dataset.gapDilation )
                return ndimage.binary_dilation(mask, structure=struc)

            sd , Method = params.WhichExperiment.Dataset.slicingInfo.slicingDim  ,  params.WhichExperiment.HardParams.Model.Method

            Dirr = params.directories.Test.Result                
            if 'train' in mode: Dirr += '/TrainData_Output'
                            
            _, Cascade_Mask = readingWithTranpose(Dirr + '/' + subject2.subjectName + '/' + params.WhichExperiment.HardParams.Model.Method.ReferenceMask + '.nii.gz' , params)
            Cascade_Mask_Dilated = dilateMask(Cascade_Mask)
            imm[Cascade_Mask_Dilated == 0] = 0
            
            return imm

        def func_Multiply_By_NotAV(imm):
            def AllNuclei__Except_AV(Dirr):                                                    
                A = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()
                Mask = []
                for cnt , nucleusName in zip(A.Indexes , A.Names):                                
                    if cnt not in [1, 2]:         
                        _, msk = readingWithTranpose(Dirr + '/' + subject2.subjectName + '/' + nucleusName + '.nii.gz' , params)
                        msk = smallFuncs.closeMask(msk > 0 , 1)
                        Mask = msk if Mask == [] else Mask + msk   

                return smallFuncs.closeMask(Mask,2) > 0.5

            Dirr = params.directories.Test.Result                
            if 'train' in mode: Dirr += '/TrainData_Output'
                
            Mask_AllExceptAV = AllNuclei__Except_AV(Dirr)
            imm[Mask_AllExceptAV] = 0

            return imm


        imF, im = readingWithTranpose(subject2.address + '/' + subject2.ImageProcessed + '.nii.gz' , params)

        if 'Cascade' in params.WhichExperiment.HardParams.Model.Method.Type and 1 not in params.WhichExperiment.Nucleus.Index and params.WhichExperiment.HardParams.Model.Method.Multiply_By_Thalmaus:
            im = func_Multiply_By_Thalmaus(im)

        if 2 in params.WhichExperiment.Nucleus.Index and params.WhichExperiment.HardParams.Model.Method.Multiply_By_Rest_For_AV:
            im = func_Multiply_By_NotAV(im)

        im = inputPreparationForUnet(im, subject2, params)
        im = normalizeA.main_normalize(params.preprocess.Normalize , im)
        im = 1 - im
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

            origMsk1N = nib.load(inputMsk).get_data() if os.path.exists(inputMsk) else np.zeros(imFshape)
            if origMsk1N.max() > 1 or origMsk1N.min() < 0: 
                print('dataset','error in label values',nameNuclei, 'min',origMsk1N.min() , 'max', origMsk1N.max() , subject.subjectName)
            origMsk1N = smallFuncs.fixMaskMinMax(origMsk1N)



            return np.expand_dims(origMsk1N ,axis=3)

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
            return (not Flag_TestOnly) and (Flag_TrainDice or Flag_cascadeMethod ) and Flag_notEmpty
        

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
            TrainList, ValList = percentageDivide(params.WhichExperiment.Dataset.Validation.percentage, list(params.directories.Train.Input.Subjects), params.WhichExperiment.Dataset.randomFlag)

            save_hdf5_subject_List(params.h5 , 'trainList' , TrainList )

            if params.WhichExperiment.Dataset.Validation.fromKeras or params.WhichExperiment.HardParams.Model.Method.Use_TestCases_For_Validation:
                DataAll.Train = separatingConcatenatingIndexes(DataAll.Train_ForTest, list(DataAll.Train_ForTest),'train')
                DataAll.Validation = ''            
            else:                
                DataAll.Train = separatingConcatenatingIndexes(DataAll.Train_ForTest, TrainList,'train')
                DataAll.Validation = separatingConcatenatingIndexes(DataAll.Train_ForTest, ValList,'validation')

                save_hdf5_subject_List(params.h5 , 'valList' , ValList )

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

            Data = {}
            g1 = params.h5.create_group(mode)
            for nameSubject, subject in tqdm(Subjects.items(), desc='Loading ' + mode):

                if ErrorInPaddingCheck(subject): continue

                im, imF = readingImage(params, subject, mode)
                origMsk , msk = readingNuclei(params, subject, imF.shape)

                msk = msk>Th
                origMsk = origMsk>Th

                if im[...,0].shape == msk[...,0].shape:
                    Data[nameSubject] = testCase(Image=im, Mask=msk ,OrigMask=(origMsk).astype('float32'), Affine=imF.get_affine(), Header=imF.get_header(), original_Shape=imF.shape)
                    saveHDf5(g1 , im , msk , origMsk , imF , subject)
                                        
                else: 
                    Error_MisMatch_In_Dim_ImageMask(subject , mode, nameSubject)


            return Data

        def readValidation(DataAll): 
            if params.WhichExperiment.HardParams.Model.Method.Use_TestCases_For_Validation:
                Read = params.WhichExperiment.Dataset.ReadTrain
                Val_Indexes = list(DataAll.Test)
            
                if (Read.Main or Read.ET or Read.CSFn) and Read.SRI: Val_Indexes = [s for s in Val_Indexes if 'SRI' not in s]

                DataAll.Validation = separatingConcatenatingIndexes(DataAll.Test, Val_Indexes, 'validation')                
                save_hdf5_subject_List(params.h5 , 'valList' , Val_Indexes)
            return DataAll
            
        DataAll = data()
        
        if trainFlag():
            DataAll.Train_ForTest = readingAllSubjects(params.directories.Train.Input.Subjects, 'train')
            DataAll = separateTrainVal_and_concatenateTrain( DataAll )


        if params.directories.Test.Input.Subjects: 
            DataAll.Test = readingAllSubjects(params.directories.Test.Input.Subjects, 'test')
            save_hdf5_subject_List(params.h5 , 'testList' , list(DataAll.Test))

            DataAll = readValidation(DataAll)

        if sagittalFlag():
            if trainFlag(): DataAll.Sagittal_Train_ForTest = readingAllSubjects(params.directories.Train.Input_Sagittal.Subjects, 'trainS')
            DataAll.Sagittal_Test          = readingAllSubjects(params.directories.Test.Input_Sagittal.Subjects , 'testS' )               
                
        return DataAll

    params = preAnalysis(params)
    print( 'InputDimensions' , params.WhichExperiment.HardParams.Model.InputDimensions )
    
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

    # def saveUserParams(params):
    #     params.UserInfo['simulation'].num_Layers = params.WhichExperiment.HardParams.Model.num_Layers
    #     params.UserInfo['InputDimensions'] = params.WhichExperiment.HardParams.Model.InputDimensions
    #     # print('InputDimensions', params.WhichExperiment.HardParams.Model.InputDimensions)
    #     # print('num_Layers', params.WhichExperiment.HardParams.Model.num_Layers)

    #     for sf in list(params.UserInfo):
    #         if '__' in sf: params.UserInfo.__delitem__(sf)

    #     smallFuncs.mkDir(params.directories.Train.Model)

    def find_PaddingValues(params):

        def findingPaddedInputSize(params):
            inputSizes = np.concatenate((params.directories.Train.Input.inputSizes , params.directories.Test.Input.inputSizes),axis=0)    
            a = 2**(params.WhichExperiment.HardParams.Model.num_Layers - 1)
            # new_inputSize = [  a * np.ceil(s / a) if s % a != 0 else s for s in np.max(inputSizes, axis=0) ]

            # new_inputSize = np.max(inputSizes, axis=0)
            # for dim in range(3):
            #     if new_inputSize[dim] % a != 0: new_inputSize[dim] = a * np.ceil(new_inputSize[dim] / a)

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
        
        if 'Cascade' in HardParams.Model.Method.Type:          
            if params.WhichExperiment.Dataset.InputPadding.Automatic: 
                inputSizes = np.concatenate((params.directories.Train.Input.inputSizes , params.directories.Test.Input.inputSizes),axis=0)
                MinInputSize = np.min(inputSizes, axis=0)
            else:
                MinInputSize = params.WhichExperiment.Dataset.InputPadding.HardDimensions

            kernel_size = HardParams.Model.Layer_Params.ConvLayer.Kernel_size.conv
            num_Layers  = HardParams.Model.num_Layers
            dim = HardParams.Model.Method.InputImage2Dvs3D
            if np.min(MinInputSize[:dim] - np.multiply( kernel_size,(2**(num_Layers - 1)))) < 0:  # ! check if the figure map size at the most bottom layer is bigger than convolution kernel size
                print('WARNING: INPUT IMAGE SIZE IS TOO SMALL FOR THE NUMBER OF LAYERS')
                num_Layers = int(np.floor( np.log2(np.min( np.divide(MinInputSize[:dim],kernel_size) )) + 1))
                print('# LAYERS  OLD:',HardParams.Model.num_Layers  ,  ' =>  NEW:',num_Layers)

            params.WhichExperiment.HardParams.Model.num_Layers = num_Layers
        return params

    params = find_AllInputSizes(params)
    params = find_correctNumLayers(params)
    params = find_PaddingValues(params)

    # saveUserParams(params)

    return params
