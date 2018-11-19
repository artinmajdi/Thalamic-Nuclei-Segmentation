import os

def whichCropMode(NucleusName, mode):
    if '1-THALAMUS' in NucleusName:
        mode = 1
    return mode

def fixDirectoryLastDashSign(dirr):
    if dirr[len(dirr)-1] == '/':
        dirr = dirr[:len(dirr)-2]

    return dirr

def checkTestDirectory(dirr):

    if 'WMnMPRAGE_bias_corr.nii.gz' in dirr:
        dd = dirr.split('/')
        for d in range(len(dd)-1):
            if d == 0:
                dirr = dd[d]
            else:
                dirr = dirr + '/' + dd[d]

        MultipleTest = 0
    else:
        if 'WMnMPRAGE_bias_corr.nii.gz' in os.listdir(dirr) :
            MultipleTest = 0
        else:
            MultipleTest = 1

        dirr = fixDirectoryLastDashSign(dirr)

    return dirr , MultipleTest

#  ---------------------- Preprocessing Params ----------------------

NucleusName = '1-THALAMUS'
SlicingDirection = 'Axial'
saveMode = 'nifti'
Augmentation = 0


Experiment_Number = 1
Directory_Tests = '/array/ssd/msmajdi/Tests/Thalamus_Keras'
Experiment_Name = 'Experiment' + str(Experiment_Number)
Directory_Experiment = Directory_Tests + '/' + Experiment_Name + '/Keras' + NucleusName.replace('-','_')
Directory_Thalamus   = Directory_Tests + '/' + Experiment_Name + '/Keras1_THALAMUS'

Directory_input = Directory_Experiment + '/train'






OptionNormalize = 1
OptionAugment = 1
OptionCrop = 1
# --cropping mode
# 1. cropping using the cropped mask acquired from rigid transformation
# 2. cropping using the cropped mask for plain size and Thalamus Prediction for slice numbers
# 3. cropping using the Thalamus prediction
CroppingMode = 2

#  ---------------------- model Params ----------------------
ArchitectureType = 'U-Net'
NumberOfLayers = 3
Optimizer = 'Adam'



# ---------------------- fixing User Defined params ----------------------
Directory_input , MultipleTest = checkTestDirectory(Directory_input)
Directory_Experiment = fixDirectoryLastDashSign(Directory_Experiment)
CroppingMode = whichCropMode(NucleusName, CroppingMode)
saveMode = saveMode.lower()
SlicingDirection = SlicingDirection.lower()
