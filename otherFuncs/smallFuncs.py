import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import nibabel as nib
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt

# TODO: Replace folder searching with "next(os.walk(directory))"
# TODO: use os.path.dirname & os.path.abspath instead of '/' remover
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

        FullIndexes = [1,2,4567,4,5,6,7,8,9,10,11,12,13,14]

    return NucleusName, FullIndexes

def listSubFolders(Dir_Prior):

    oldStandard = 1
    if oldStandard == 1:
        subFolders = []
        subFlds = os.listdir(Dir_Prior)
        for i in range(len(subFlds)):
            if subFlds[i][:5] == 'vimp2':
                subFolders.append(subFlds[i])
    else:
        subFolders = os.listdir(Dir_Prior)

    return subFolders

def mkDir(Dir):
    if not os.path.isdir(Dir):
        os.makedirs(Dir)
    return Dir

def choosingSubject(Input):
    return Input.Image.get_data() , Input.CropMask.get_data() , Input.ThalamusMask.get_data() , Input.TestAddress

def saveImage(Image , Affine , Header , outDirectory):
    out = nib.Nifti1Image(Image,Affine)
    out.get_header = Header
    nib.save(out , outDirectory)

def terminalEntries(params):

    for en in range(len(sys.argv)):
        entry = sys.argv[en]

        if entry.lower() == '-g':  # gpu num
            params.directories.Experiment.HardParams.Machine.GPU_Index = sys.argv[en+1]

        elif entry.lower() == '-o':  # output directory
            params.directories.Train.Model = mkDir(sys.argv[en+1] + '/Experiment' + str(params.directories.Experiment.Experiment_Index) + '/models'  + '/SubExperiment' + str(params.directories.Experiment.SubExperiment_Index) + '_' + params.directories.Experiment.Tag)
            params.directories.Test.Result = mkDir(sys.argv[en+1] + '/Experiment' + str(params.directories.Experiment.Experiment_Index) + '/Results' + '/SubExperiment' + str(params.directories.Experiment.SubExperiment_Index) + '_' + params.directories.Experiment.Tag)

        elif entry.lower() == '-m':  # which machine; server localPC local Laptop
            params.directories.Experiment.HardParams.Machine.WhichMachine = sys.argv[en+1]

        elif entry.lower() == '-n':  # nuclei index
            if sys.argv[en+1].lower() == 'all':
                params.directories.Experiment.Nucleus.Index = np.append([1,2,4567],range(4,14))

            elif sys.argv[en+1][0] == '[':
                B = sys.argv[en+1].split('[')[1].split(']')[0].split(",")
                params.directories.Experiment.Nucleus.Index = [int(k) for k in B]

            else:
                params.directories.Experiment.Nucleus.Index = [int(sys.argv[en+1])]

            params.directories.Experiment.Nucleus.Name = "check the indexes entered by user!"

        elif entry.lower() == '-i': # input image or directory
            params.directories.Experiment.Address = sys.argv[en+1]
            params.directories.Train.Address =  sys.argv[en+1] + '/Experiment' + str(params.directories.Experiment.Experiment_Index) + '/Train'
            params.directories.Train.Input   = checkInputDirectory( params.directories.Train.Address ,params.directories.Experiment.Nucleus.Name)
            params.directories.Test.Address  =  sys.argv[en+1] + '/Experiment' + str(params.directories.Experiment.Experiment_Index) + '/Test'
            params.directories.Test.Input    = checkInputDirectory( params.directories.Test.Address ,params.directories.Experiment.Nucleus.Name)

        elif entry.lower() == '-TemplateMask':  # template Mask
            params.directories.Experiment.HardParams.Template.Mask = sys.argv[en+1]

    return params

def checkInputDirectory(Dir,NucleusName):

    # multipleTest , files , subfolders = checkMultipleTestOrNot(Dir,NucleusName)

    subjects = {}
    for sf in os.listdir(Dir):
        subjects[sf] = InputNames(Dir + '/' + sf ,NucleusName)

    if len(subjects) == 1:
        multipleTest = False
    else:
        multipleTest = True

    class Input:
        Address = fixDirectoryLastDashSign(Dir)
        Subjects = subjects
        MultipleTest = multipleTest

    return Input

def checkMultipleTestOrNot(Dir,NucleusName):

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

def funcExpDirectories(experiment):

    class train:
        Address = experiment.Address + '/Experiment' + str(experiment.Experiment_Index) + '/Train'
        Model   = mkDir(experiment.Address + '/Experiment' + str(experiment.Experiment_Index) + '/models'  + '/SubExperiment' + str(experiment.SubExperiment_Index) + '_' + experiment.Tag)
        Input   = checkInputDirectory( Address ,experiment.Nucleus.Name)

    class test:
        Address = experiment.Address + '/Experiment' + str(experiment.Experiment_Index) + '/Test'
        Result  = mkDir(experiment.Address + '/Experiment' + str(experiment.Experiment_Index) + '/Results' + '/SubExperiment' + str(experiment.SubExperiment_Index) + '_' + experiment.Tag)
        Input   = checkInputDirectory( Address ,experiment.Nucleus.Name)

    class Directories:
        Experiment = experiment
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
        Address = ''
        testWarp = ''
        testInverseWarp = ''
        testAffine = ''

    class temp:
        CropMask = ''
        Cropped = ''
        BiasCorrected = ''
        Deformation = deformation
        Address = ''

    class tempLabel:
        Address = ''
        Cropped = ''

    class label:
        LabelProcessed = ''
        LabelOriginal = ''
        Temp = tempLabel
        Address = ''

    class Files:
        ImageOriginal = '' # WMn_MPRAGE'
        ImageProcessed = ''
        Label = label
        Temp = temp
        Address = Dir



    Files.Label.Address =  ''
    flagTemp = False
    for d in os.listdir(Dir):
        if '.nii.gz' in d:
            flagTemp = True
            if '_PProcessed.nii.gz' in d:
                Files.ImageProcessed = d.split('.nii.gz')[0]
            else:
                Files.ImageOriginal = d.split('.nii.gz')[0]
        elif 'temp' not in d:
                Files.Label.Address = Dir + '/' + d

    if flagTemp:
        Files.Temp.Address = mkDir(Dir + '/temp')
        Files.Temp.Deformation.Address = mkDir(Dir + '/temp/deformation')

    if os.path.exists(Files.Label.Address):

        Files.Label.Temp.Address =  mkDir(Files.Label.Address + '/temp')
        for d in os.listdir(Files.Label.Address):
            if NucleusName + '.nii.gz' in d:
                Files.Label.LabelOriginal = d.split('.nii.gz')[0]
            elif NucleusName + '_PProcessed.nii.gz' in d:
                Files.Label.LabelProcessed = d.split('.nii.gz')[0]

            elif 'temp' in d:
                Files.Label.Temp.Address = Files.Label.Address + '/' + d

                for d in os.listdir(Files.Label.Temp.Address):
                    if '_Cropped.nii.gz' in d:
                        Files.Label.Temp.Cropped = d.split('.nii.gz')[0]

    for d in os.listdir(Files.Temp.Address):

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
            Files.Temp.Deformation.Address = Files.Temp.Address + '/' + d

            for d in os.listdir( Files.Temp.Deformation.Address ):
                if 'testWarp.nii.gz' in d:
                    Files.Temp.Deformation.testWarp = d.split('.nii.gz')[0]
                elif 'testInverseWarp.nii.gz' in d:
                    Files.Temp.Deformation.testInverseWarp = d.split('.nii.gz')[0]
                elif 'testAffine.txt' in d:
                    Files.Temp.Deformation.testAffine = d.split('.nii.gz')[0]


    return Files

def inputNamesCheck(params):

    for mode in ['Train' , 'Test']:

        if params.preprocess.TestOnly and 'Train' in mode:
            continue

        dirr = params.directories.Train if 'Train' in mode else params.directories.Test

        for sj in dirr.Input.Subjects:
            subject = dirr.Input.Subjects[sj]

            if params.preprocess.Debug.PProcessExist:

                files = os.listdir(subject.Address)

                flagPPExist = False
                for si in files:
                    if '_PProcessed' in si:
                        flagPPExist = True
                        break

                if not flagPPExist:
                    sys.exit('preprocess files doesn\'t exist ' + 'Subject: ' + sj + ' Dir: ' + subject.Address)

            else: # if not params.preprocess.Debug.PProcessExist and

                imOrig = subject.Address + '/' + subject.ImageOriginal + '.nii.gz'
                imProc = subject.Address + '/' + subject.ImageOriginal + '_PProcessed.nii.gz'
                copyfile( imOrig  , imProc )

                for ind in params.directories.Experiment.Nucleus.FullIndexes:
                    NucleusName, _ = NucleiSelection(ind , params.directories.Experiment.Nucleus.Organ)

                    mskOrig = subject.Label.Address + '/' + NucleusName + '.nii.gz'
                    mskProc = subject.Label.Address + '/' + NucleusName + '_PProcessed.nii.gz'
                    copyfile( mskOrig , mskProc)

    params.directories = funcExpDirectories(params.directories.Experiment)
    return params

def inputSizes(Subjects):
    inputSize = []
    for sj in Subjects:
        inputSize.append( nib.load(Subjects[sj].Address + '/' + Subjects[sj].ImageProcessed + '.nii.gz').shape )

    return np.array(inputSize)

def correctNumLayers(Subjects, HardParams):

    inputSize = inputSizes(Subjects)

    MinInputSize = np.min(inputSize, axis=0)
    kernel_size = HardParams.Model.ConvLayer.Kernel_size.conv
    num_Layers  = HardParams.Model.num_Layers

    if np.min(  MinInputSize[:2] - np.multiply( kernel_size,(2**(num_Layers - 1)) )  ) < 0:  # ! check if the figure map size at the most bottom layer is bigger than convolution kernel size
        print('WARNING: INPUT IMAGE SIZE IS TOO SMALL FOR THE NUMBER OF LAYERS')
        num_Layers = int(np.floor( np.log2(np.min( np.divide(MinInputSize[:2],kernel_size) )) + 1 ))
        print('# LAYERS  OLD:',HardParams.Model.num_Layers  ,  ' =>  NEW:',num_Layers)

    HardParams.Model.num_Layers = num_Layers
    return HardParams

def imageSizesAfterPadding(Subjects, HardParams):

    inputSize = inputSizes(Subjects)

    #! Finding the final image sizes after padding
    MaxInputSize = np.max(inputSize, axis=0)
    new_inputSize = MaxInputSize.copy()
    a = 2**(HardParams.Model.num_Layers - 1)
    for dim in range(2):
        # checking how much we need to padd the input image to make sure the we don't lose any information because of odd dimension sizes
        if MaxInputSize[dim] % a != 0:
            new_inputSize[dim] = a * np.ceil(MaxInputSize[dim] / a)


    # ! finding the amount of padding for each subject in each direction
    fullpadding = new_inputSize[:2] - inputSize[:,:2]
    md = np.mod(fullpadding,2)

    for sn, name in enumerate(list(Subjects)):
        padding = [np.zeros(2)]*3
        for dim in range(2):
            if md[sn,dim] == 0:
                padding[dim] = tuple([int(fullpadding[sn,dim]/2)]*2)
            else:
                padding[dim] = tuple([int(np.floor(fullpadding[sn,dim]/2) + 1) , int(np.floor(fullpadding[sn,dim]/2))])

        padding[2] = tuple([0,0])
        Subjects[name].Padding = tuple(padding)


    HardParams.Model.InputDimensions = new_inputSize
    return Subjects, HardParams

def imShow(*args):

    _, axes = plt.subplots(1,len(args))
    for sh in range(len(args)):
        axes[sh].imshow(np.squeeze(args[sh]),cmap='gray')

    plt.show()

    return True
