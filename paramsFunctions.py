import os
from collections import namedtuple

def whichCropMode(NucleusName, mode):
    if '1-THALAMUS' in NucleusName:
        mode = 1
    return mode

def fixDirectoryLastDashSign(dirr):
    if dirr[len(dirr)-1] == '/':
        dirr = dirr[:len(dirr)-2]

    return dirr

def augmentLengthChecker(augment,augmentLength):
    if not augment:
        augmentLength = 0

    return augmentLength

def checkTestDirectory(dirr):

    if 'WMnMPRAGE_bias_corr.nii.gz' in dirr:
        dd = dirr.split('/')
        for d in range(len(dd)-1):
            if d == 0:
                dirr = dd[d]
            else:
                dirr = dirr + '/' + dd[d]

        MultipleTest = 'False'
    else:
        if 'WMnMPRAGE_bias_corr.nii.gz' in os.listdir(dirr) :
            MultipleTest = 'False'
            dirr = fixDirectoryLastDashSign(dirr)
        else:
            MultipleTest = 'True'

    return dirr , MultipleTest

def funcExpDirectories(Experiment_Number , subExperiment_Number):

    # struct2 = namedtuple('struct' , 'AllExperiments Train Test Results models ThalamusPrediction Input Output')

    class Directories:
        AllExperiments = '/array/ssd/msmajdi/Tests/Thalamus_Keras'
        Train   = AllExperiments + '/Experiment' + str(Experiment_Number) + '/' + 'Train'
        Test    = AllExperiments + '/Experiment' + str(Experiment_Number) + '/' + 'Test'
        Results = AllExperiments + '/Experiment' + str(Experiment_Number) + '/' + 'Results' + '/subExperiment' + str(Experiment_Number)
        models  = AllExperiments + '/Experiment' + str(Experiment_Number) + '/' + 'models'  + '/subExperiment' + str(Experiment_Number)

        ThalamusPrediction = Results + '/1-THALAMUS.nii.gz'

    # Directories = struct2(Dir_AllExperiments , Dir_Train , Dir_Test , Dir_Results , Dir_models , Dir_Thalamus , "" , "")

    return Directories

    