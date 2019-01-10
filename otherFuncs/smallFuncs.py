import nibabel as nib
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import os, sys
import mat4py
import pickle

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# TODO: Replace folder searching with "next(os.walk(directory))"
# TODO: use os.path.dirname & os.path.abspath instead of '/' remover
# TODO: sort the input images so that it is system independent
def NucleiSelection(ind = 1,organ = 'THALAMUS'):

    if 'THALAMUS' in organ:
        if ind == 1:
            NucleusName = '1-THALAMUS'
        elif ind == 2:
            NucleusName = '2-AV'
        elif ind == 4567:
            NucleusName = '4567-VL'
        elif ind == 4:
            NucleusName = '4-VA'
        elif ind == 5:
            NucleusName = '5-VLa'
        elif ind == 6:
            NucleusName = '6-VLP'
        elif ind == 7:
            NucleusName = '7-VPL'
        elif ind == 8:
            NucleusName = '8-Pul'
        elif ind == 9:
            NucleusName = '9-LGN'
        elif ind == 10:
            NucleusName = '10-MGN'
        elif ind == 11:
            NucleusName = '11-CM'
        elif ind == 12:
            NucleusName = '12-MD-Pf'
        elif ind == 13:
            NucleusName = '13-Hb'
        elif ind == 14:
            NucleusName = '14-MTT'

        FullIndexes = [1,2,5,6,7,8,9,10,11,12,13,14]

    return NucleusName, FullIndexes

def listSubFolders(Dir):

    oldStandard = True

    Dir_Prior = next(os.walk(Dir))[1]
    subFolders = []
    if len(Dir_Prior) > 0:
        
        if oldStandard:
            for subFlds in Dir_Prior:
                if 'vimp' in subFlds: subFolders.append(subFlds)
        else:
            subFolders = Dir_Prior

    return subFolders

def mkDir(Dir):
    if not os.path.isdir(Dir): os.makedirs(Dir)
    return Dir

def choosingSubject(Input):
    return Input.Image.get_data() , Input.CropMask.get_data() , Input.ThalamusMask.get_data() , Input.TestAddress

def saveImage(Image , Affine , Header , outDirectory):
    out = nib.Nifti1Image((Image).astype('float32'),Affine)
    out.get_header = Header
    nib.save(out , outDirectory)

def terminalEntries( UserInfo):

    for en in range(len(sys.argv)):
        entry = sys.argv[en]

        if entry.lower() in ('-g','--gpu'):  # gpu num
            UserInfo.GPU_Index = sys.argv[en+1]

        elif entry.lower() in ('-n','--nuclei'):  # nuclei index
            if sys.argv[en+1].lower() == 'all':
                UserInfo.nucleus_Index = np.append([1,2,4567],range(4,14))

            elif sys.argv[en+1][0] == '[':
                B = sys.argv[en+1].split('[')[1].split(']')[0].split(",")
                UserInfo.nucleus_Index = [int(k) for k in B]

            else:
                UserInfo.nucleus_Index = [int(sys.argv[en+1])]

        elif entry.lower() in ('-l','--loss'):
            UserInfo.lossFunctionIx = int(sys.argv[en+1])

        elif entry.lower() in ('-d','--dataset'):
            UserInfo.DatasetIx = int(sys.argv[en+1])

        elif entry.lower() in ('-e','--epochs'):
            UserInfo.epochs = int(sys.argv[en+1])

        elif entry.lower() in ('-sIx','--SubExperiment_Index'):
            UserInfo.SubExperiment_Index = int(sys.argv[en+1])

        elif entry.lower() in ('-Ix','--Experiments_Index'):
            UserInfo.Experiments_Index = int(sys.argv[en+1])

    return UserInfo


def checkInputDirectory(Dir, NucleusName):

    # multipleTest , files , subfolders = checkMultipleTestOrNot(Dir,NucleusName)

    subjects = {}

    for sf in listSubFolders(Dir): # os.listdir(Dir):
        subjects[sf] = InputNames(Dir + '/' + sf ,NucleusName)

    if len(subjects) == 1:
        multipleTest = False
    else:
        multipleTest = True

    class Input:
        address = fixDirectoryLastDashSign(Dir)
        Subjects = subjects
        MultipleTest = multipleTest

    return Input

def checkMultipleTestOrNot(Dir, NucleusName):

    subjects = ''
    files = ''

    if '.nii.gz' in os.path.basename(Dir):
        # dd = Dir.split('/')
        # Dir = ''
        # for d in range(len(dd)-1):
        #     Dir = Dir + dd[d] + '/'
        Dir = Dir.split(os.path.basename(Dir))[0]

        files = InputNames(Dir ,NucleusName)
        multipleTest = 'False'
    else:
        subjects = os.listdir(Dir)

        flag = False
        for ss in subjects:
            if '.nii.gz' in ss:
                flag = True
                break

        if flag or len(subjects) == 1:
            multipleTest = 'False'
            files = InputNames(Dir,NucleusName)
        else:
            multipleTest = 'True'

    return multipleTest , files , subjects

def funcExpDirectories(whichExperiment):

    class train:
        address = mkDir(whichExperiment.Experiment.address + '/train')
        Model   = mkDir(whichExperiment.Experiment.address + '/models/' + whichExperiment.SubExperiment.name + '/' + whichExperiment.Nucleus.name)
        Model_Thalamus   = whichExperiment.Experiment.address + '/models/' + whichExperiment.SubExperiment.name_thalamus + '/1-THALAMUS'
        Input   = checkInputDirectory(address, whichExperiment.Nucleus.name)

    class test:
        address = mkDir(whichExperiment.Experiment.address + '/test')
        Result  = mkDir(whichExperiment.Experiment.address + '/results/' + whichExperiment.SubExperiment.name)
        Input   = checkInputDirectory(address, whichExperiment.Nucleus.name)

    class Directories:
        # WhichExperiment = whichExperiment
        Train = train
        Test  = test

    return Directories

def whichCropMode(NucleusName, mode):
    if '1-THALAMUS' in NucleusName:
        mode = 1
    return mode

def fixDirectoryLastDashSign(Dir):
    Dir = os.path.abspath(Dir)
    # if Dir[len(Dir)-1] == '/':
    #     Dir = Dir[:len(Dir)-2]

    return Dir

def augmentLengthChecker(augment):
    if not augment.Mode:
        augment.AugmentLength = 0

    return augment

def InputNames(Dir , NucleusName):

    class deformation:
        address = ''
        testWarp = ''
        testInverseWarp = ''
        testAffine = ''

    class temp:
        CropMask = ''
        Cropped = ''
        BiasCorrected = ''
        Deformation = deformation
        address = ''

    class tempLabel:
        address = ''
        Cropped = ''

    class label:
        LabelProcessed = ''
        LabelOriginal = ''
        Temp = tempLabel
        address = ''

    class Files:
        ImageOriginal = '' # WMn_MPRAGE'
        ImageProcessed = ''
        Label = label
        Temp = temp
        address = Dir



    Files.Label.address = ''
    flagTemp = False
    for d in os.listdir(Dir):
        if '.nii.gz' in d:
            flagTemp = True
            if '_PProcessed.nii.gz' in d:
                Files.ImageProcessed = d.split('.nii.gz')[0]
            else:
                Files.ImageOriginal = d.split('.nii.gz')[0]
        elif 'temp' not in d:
            Files.Label.address = Dir + '/' + d

    if flagTemp:
        Files.Temp.address = mkDir(Dir + '/temp')
        Files.Temp.Deformation.address = mkDir(Dir + '/temp/deformation')

    if os.path.exists(Files.Label.address):

        Files.Label.Temp.address =  mkDir(Files.Label.address + '/temp')
        for d in os.listdir(Files.Label.address):
            if NucleusName + '.nii.gz' in d:
                Files.Label.LabelOriginal = d.split('.nii.gz')[0]
            elif NucleusName + '_PProcessed.nii.gz' in d:
                Files.Label.LabelProcessed = d.split('.nii.gz')[0]

            elif 'temp' in d:
                Files.Label.Temp.address = Files.Label.address + '/' + d

                for d in os.listdir(Files.Label.Temp.address):
                    if '_Cropped.nii.gz' in d:
                        Files.Label.Temp.Cropped = d.split('.nii.gz')[0]

    for d in os.listdir(Files.Temp.address):

        if '.nii.gz' in d:
            if 'CropMask.nii.gz' in d:
                Files.Temp.CropMask = d.split('.nii.gz')[0]
            elif '_bias_corr.nii.gz' in d:
                Files.Temp.BiasCorrected = d.split('.nii.gz')[0]
            elif '_bias_corr_Cropped.nii.gz' in d:
                Files.Temp.Cropped = d.split('.nii.gz')[0]
            else:
                Files.Temp.origImage = d.split('.nii.gz')[0]

        elif 'deformation' in d:
            Files.Temp.Deformation.address = Files.Temp.address + '/' + d

            for d in os.listdir(Files.Temp.Deformation.address):
                if 'testWarp.nii.gz' in d:
                    Files.Temp.Deformation.testWarp = d.split('.nii.gz')[0]
                elif 'testInverseWarp.nii.gz' in d:
                    Files.Temp.Deformation.testInverseWarp = d.split('.nii.gz')[0]
                elif 'testAffine.txt' in d:
                    Files.Temp.Deformation.testAffine = d.split('.nii.gz')[0]


    return Files

# TODO fix "inputNamesCheck" function to count for situations when we only want to apply the function on one case
def inputNamesCheck(params, mode):

    if 'experiment' in mode:
        for wFolder in ['Train' , 'Test']:

            if params.preprocess.TestOnly and 'Train' in wFolder:
                continue

            dirr = params.directories.Train if 'Train' in wFolder else params.directories.Test

            for sj in dirr.Input.Subjects:
                subject = dirr.Input.Subjects[sj]

                if params.preprocess.Debug.PProcessExist:

                    files = os.listdir(subject.address)

                    flagPPExist = False
                    for si in files:
                        if '_PProcessed' in si:
                            flagPPExist = True
                            break

                    if not flagPPExist:
                        sys.exit('preprocess files doesn\'t exist ' + 'Subject: ' + sj + ' Dir: ' + subject.address)

                else: # if not params.preprocess.Debug.PProcessExist and

                    imOrig = subject.address + '/' + subject.ImageOriginal + '.nii.gz'
                    imProc = subject.address + '/' + subject.ImageOriginal + '_PProcessed.nii.gz'
                    copyfile(imOrig  , imProc)

                    for ind in params.WhichExperiment.Nucleus.FullIndexes:
                        NucleusName, _ = NucleiSelection(ind, params.WhichExperiment.Nucleus.Organ)

                        mskOrig = subject.Label.address + '/' + NucleusName + '.nii.gz'
                        mskProc = subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz'
                        copyfile(mskOrig, mskProc)

        params.directories = funcExpDirectories(params.WhichExperiment)

    else:
        print('')

    return params

def inputSizes(Subjects):
    inputSize = []
    for sj in Subjects:
        inputSize.append( nib.load(Subjects[sj].address + '/' + Subjects[sj].ImageProcessed + '.nii.gz').shape)

    return np.array(inputSize)

def correctNumLayers(params):

    HardParams = params.WhichExperiment.HardParams

    inputSize = inputSizes(params.directories.Train.Input.Subjects)

    MinInputSize = np.min(inputSize, axis=0)
    kernel_size = HardParams.Model.ConvLayer.Kernel_size.conv
    num_Layers  = HardParams.Model.num_Layers

    if np.min(MinInputSize[:2] - np.multiply( kernel_size,(2**(num_Layers - 1)))) < 0:  # ! check if the figure map size at the most bottom layer is bigger than convolution kernel size
        print('WARNING: INPUT IMAGE SIZE IS TOO SMALL FOR THE NUMBER OF LAYERS')
        num_Layers = int(np.floor( np.log2(np.min( np.divide(MinInputSize[:2],kernel_size) )) + 1))
        print('# LAYERS  OLD:',HardParams.Model.num_Layers  ,  ' =>  NEW:',num_Layers)

    params.WhichExperiment.HardParams.Model.num_Layers = num_Layers
    return params

def imageSizesAfterPadding(params, mode):

    if 'experiment' in mode:
        for wFolder in ['Train' , 'Test']:

            if params.preprocess.TestOnly and 'Train' in wFolder:
                continue

            Subjects = params.directories.Train.Input.Subjects if 'Train' in wFolder else params.directories.Test.Input.Subjects
            inputSize = inputSizes(Subjects)

            #! Finding the final image sizes after padding
            if 'Train' in wFolder:
                MaxInputSize = np.max(inputSize, axis=0)
                new_inputSize = MaxInputSize

                a = 2**(params.WhichExperiment.HardParams.Model.num_Layers - 1)
                for dim in range(2):
                    # checking how much we need to pad the input image to make sure the we don't lose any information because of odd dimension sizes
                    if MaxInputSize[dim] % a != 0:
                        new_inputSize[dim] = a * np.ceil(MaxInputSize[dim] / a)

                params.WhichExperiment.HardParams.Model.InputDimensions = new_inputSize
            else:
                new_inputSize = params.WhichExperiment.HardParams.Model.InputDimensions


            #! finding the amount of padding for each subject in each direction
            fullpadding = new_inputSize[:2] - inputSize[:,:2]
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

            if 'Train' in wFolder:
                params.directories.Train.Input.Subjects = Subjects
            else:
                params.directories.Test.Input.Subjects = Subjects

    return params

# TODO check the matlab imshow3D see if i can use it in python
def imShow(*args):

    _, axes = plt.subplots(1,len(args))
    for sh in range(len(args)):
        axes[sh].imshow(np.squeeze(args[sh]),cmap='gray')

    plt.show()

    return True

def unPadding(im , pad):
    sz = im.shape
    return im[pad[0][0]:sz[0]-pad[0][1] , pad[1][0]:sz[1]-pad[1][1] , pad[2][0]:sz[2]-pad[2][1]]

def Dice_Calculator(msk1,msk2):
    intersection = msk1*msk2
    return intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)


def saveReport(DirSave, name , data, method):

    if 'pickle' in method:
        savePickle(DirSave + '/' + name + '.pkl', data)
    elif 'mat' in method:
        mat4py.savemat(DirSave + '/' + name + '.mat', data)

def loadReport(DirSave, name, method):

    if 'pickle' in method:
        return loadPickle(DirSave + '/' + name + '.pkl')
    elif 'mat' in method:
        return mat4py.loadmat(DirSave + '/' + name + '.pkl')

#! saving the user parameters
def Saving_UserInfo(DirSave, params, UserInfo):

    UserInfo.InputDimensions = str(params.WhichExperiment.HardParams.Model.InputDimensions)
    UserInfo.num_Layers      = params.WhichExperiment.HardParams.Model.num_Layers

    saveReport(DirSave, 'UserInfo', dict_from_module(UserInfo) , UserInfo.SaveReportMethod)

def Loading_UserInfo(DirLoad, method):
    UserInfo = loadReport(DirLoad, 'UserInfo', method)
    UserInfo = dict2obj( UserInfo )

    a = UserInfo.InputDimensions.replace(',' ,'').split('[')[1].split(']')[0].split(' ')
    UserInfo.InputDimensions = [int(ai) for ai in a]


    return UserInfo

def savePickle(Dir, data):
    f = open(Dir,"wb")
    pickle.dump(data,f)
    f.close()

def loadPickle(Dir):
    f = open(Dir,"wb")
    data = pickle.load(f)
    f.close()
    return data


def dict_from_module(module):
    context = {}
    for setting in dir(module):
        if '__' not in setting:
            context[setting] = getattr(module, setting)

    return context

def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

def gpuConfig(GPU_Index):
    import os
    import tensorflow as tf
    from keras import backend as K
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_Index
    session = tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   )
    K.set_session(session)
    return K