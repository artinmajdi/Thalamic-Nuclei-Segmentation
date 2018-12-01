import os
import nibabel as nib
import numpy as np
import sys
from collections import namedtuple
from shutil import copyfile

def NucleiSelection(ind):

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

    return NucleusName

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

    if '.nii.gz' in Dir:
        dd = Dir.split('/')
        Dir = ''
        for d in range(len(dd)-1):
            Dir = Dir + dd[d] + '/'

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
    if Dir[len(Dir)-1] == '/':
        Dir = Dir[:len(Dir)-2]

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
            if '_PP.nii.gz' in d:
                Files.ImageProcessed = d.split('.nii.gz')[0]           
            else:
                Files.ImageOriginal = d.split('.nii.gz')[0]
        elif 'temp' not in d:             
                Files.Label.Address = Dir + '/' + d 

    if flagTemp:
        Files.Temp.Address = mkDir(Dir + '/temp')

    if os.path.exists(Files.Label.Address):

        Files.Label.Temp.Address =  mkDir(Files.Label.Address + '/temp')
        for d in os.listdir(Files.Label.Address):
            if NucleusName + '.nii.gz' in d:
                Files.Label.LabelOriginal = d.split('.nii.gz')[0]
            elif NucleusName + '_PP.nii.gz' in d:
                Files.Label.LabelProcessed = d.split('.nii.gz')[0]           
                    
            elif 'temp' in d:
                Files.Label.Temp.Address = Files.Label.Address + '/' + d


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

def inputNamesCheck(params,mode):

    dirr = params.directories.Train if mode == 'Train' else params.directories.Test
    for sj in dirr.Input.Subjects:
        subject = dirr.Input.Subjects[sj]

        imOrig = subject.Address + '/' + subject.ImageOriginal + '.nii.gz'
        mskOrig = subject.Label.Address + '/' + subject.Label.LabelOriginal + '.nii.gz'

        if subject.ImageProcessed:
            imProc = subject.Address + '/' + subject.ImageProcessed + '.nii.gz'
            mskProc = subject.Label.Address + '/' + subject.Label.LabelProcessed + '.nii.gz'
        else:
            imProc = subject.Address + '/' + subject.ImageOriginal + '_PP.nii.gz'
            mskProc = subject.Label.Address + '/' + subject.Label.LabelOriginal + '_PP.nii.gz'

        if '_Aug' not in sj:
            if params.preprocess.Mode:
                copyfile( imOrig , imProc)
                copyfile( mskOrig , mskProc)

            elif (os.path.isfile(imOrig)) and (not os.path.isfile(imProc)):
                copyfile(imOrig , imProc)            
                copyfile( mskOrig , mskProc)


    params.directories = funcExpDirectories(params.directories.Experiment)
    return params
    
