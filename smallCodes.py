import os
import nibabel as nib
import numpy as np
import sys
from collections import namedtuple

# a = funcExpDirectories(1,1)
# b = a()

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
    try:
        os.stat(Dir)
    except:
        os.makedirs(Dir)
    return Dir

def choosingSubject(Input):
    return Input.Image.get_data() , Input.CropMask.get_data() , Input.ThalamusMask.get_data() , Input.TestAddress

def saveImage(Image , Affine , Header , outDirectory):
    out = nib.Nifti1Image(Image,Affine)
    out.get_header = Header
    nib.save(out , outDirectory)

def saveMain(Input):

    outDirectory = ''
    for AugIx in Input.Image.shape[3]:
        saveImage(Input.Image[...,AugIx] , Input.Affine , Input.Header , outDirectory)
        saveImage(Input.CropMask[...,AugIx] , Input.Affine , Input.Header , outDirectory)
        saveImage(Input.ThalamusMask[...,AugIx] , Input.Affine , Input.Header , outDirectory)

def terminalEntries(params):

    params.gpuNum =  '4'  # 'nan'  #
    params.IxNuclei = [1]
    params.whichMachine = 'server'

    for en in range(len(sys.argv)):
        entry = sys.argv[en]

        if entry.lower() == '-g':  # gpu num
            params.gpuNum = sys.argv[en+1]

        elif entry.lower() == '-i': # input image or directory
            params.directories.Input = checkInputDirectory( sys.argv[en+1] )

        elif entry.lower() == '-o':  # output directory
            params.directories.Output = sys.argv[en+1]

        elif entry.lower() == '-m':  # which machine; server localPC local Laptop
            params.whichMachine = sys.argv[en+1]

        elif entry.lower() == '-n':  # nuclei index
            if sys.argv[en+1].lower() == 'all':
                params.IxNuclei = np.append([1,2,4567],range(4,14))

            elif sys.argv[en+1][0] == '[':
                B = sys.argv[en+1].split('[')[1].split(']')[0].split(",")
                params.IxNuclei = [int(k) for k in B]

            else:
                params.IxNuclei = [int(sys.argv[en+1])]

        elif entry.lower() == '-TemplateMask':  # template Mask
            params.directories.TemplateMask = sys.argv[en+1]

    return params

def checkInputDirectory(Dir):

    if '.nii.gz' in Dir:
        dd = Dir.split('/')
        Dir = ''
        for d in range(len(dd)-1):
            Dir = Dir + dd[d] + '/'

        files = InputNames(Dir)
        multipleTest = 'False'
    else:
        subfiles = os.listdir(Dir)

        flag = 0
        for ss in subfiles:
            if '.nii.gz' in ss:
                flag = 1
                break

        if flag == 1:
            multipleTest = 'False'
            files = InputNames(Dir)
        else:
            files = subfiles
            multipleTest = 'True'

    class Input:
        Address = fixDirectoryLastDashSign(Dir)
        Files = files
        MultipleTest = multipleTest

    return Input

def checkOutputDirectory(user , subExperiment_Number):

    class Output:
        Address  = user.output
        Result   = mkDir( user.output + '/' + 'Results' + '/subExperiment' + str(subExperiment_Number) )
        Model    = mkDir( user.output + '/' + 'models'  + '/subExperiment' + str(subExperiment_Number) )
        Thalamus = user.Thalamus

    return Output

def funcExpDirectories(Experiment_Number , subExperiment_Number):

    # struct2 = namedtuple('struct' , 'AllExperiments Train Test Results models ThalamusPrediction Input Output')

    AllExperiments = '/array/ssd/msmajdi/experiments/Keras'

    class Experiment:
        Train = AllExperiments + '/Experiment' + str(Experiment_Number) + '/' + 'Train'
        Test  = AllExperiments + '/Experiment' + str(Experiment_Number) + '/' + 'Test'

    class Template:
        Image = '/array/ssd/msmajdi/code/RigidRegistration' + '/origtemplate.nii.gz'
        Mask = '/array/ssd/msmajdi/code/RigidRegistration' + '/MyCrop_Template2_Gap20.nii.gz'

    class Inf:
        output = AllExperiments
        Thalamus = AllExperiments + '/' + 'Results' + '/subExperiment' + str(subExperiment_Number) + '/1-THALAMUS.nii.gz' 

    class Directories:
        Experiment
        Output = checkOutputDirectory(Inf , subExperiment_Number)
        Input  = checkInputDirectory( Experiment.Train )
        Template

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
    if not augment.mode:
        augment.augmentLength = 0

    return augment

def InputNames(Dir):

    class Files:
        Crop = ''
        BiasCorrected = ''
        origImage = ''

    for d in os.listdir(Dir):
        if '.nii.gz' in d:
            if '_Crop.nii.gz' in d:
                Files.Crop = d.split('.nii.gz')[0]
            elif '_bias_corr.nii.gz' in d:
                Files.BiasCorrected = d.split('.nii.gz')[0]
            else:
                Files.origImage = d.split('.nii.gz')[0]

    return Files
