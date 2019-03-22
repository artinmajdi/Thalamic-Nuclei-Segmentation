import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import otherFuncs.smallFuncs as smallFuncs
from shutil import copyfile

def RigidRegistration(subject , Template , preprocess):

    print('     Rigid Registration')
    processed = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.Temp.address + '/CropMask.nii.gz'
    LinearAffine = subject.Temp.Deformation.address + '/linearAffine.txt'
    if preprocess.Mode and preprocess.Cropping.Mode:

        if not os.path.isfile(outP):
            if not os.path.isfile(LinearAffine):
                os.system("ANTS 3 -m CC[%s, %s ,1,5] -o %s -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(processed , Template.Image , subject.Temp.Deformation.address + '/linear') )

            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Mask , outP , processed , LinearAffine) )

def BiasCorrection(subject , params):

    print('     Bias Correction')

    inP  = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_bias_corr.nii.gz'
    if params.preprocess.Mode and params.preprocess.BiasCorrection.Mode:

        if os.path.isfile(outDebug) and params.preprocess.Debug.justForNow:
            copyfile(outDebug , outP)
        else:
            os.system( "N4BiasFieldCorrection -d 3 -i %s -o %s -b [200] -s 3 -c [50x50x30x20,1e-6]"%( inP, outP )  )
            if params.preprocess.Debug.doDebug:
                copyfile(outP , outDebug)

# def Bash_Cropping(subject , params):

#     if params.preprocess.Cropping.Mode:

#         inP  = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
#         outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
#         crop = subject.Temp.address + '/CropMask.nii.gz'
#         outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_Cropped.nii.gz'

#         if os.path.isfile(outDebug) and params.preprocess.Debug.justForNow:
#             copyfile(outDebug , outP)
#         else:
#             os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )
#             if params.preprocess.Debug.doDebug:
#                 copyfile(outP , outDebug)

#         # Cropping the Label
#         for ind in params.WhichExperiment.Nucleus.FullIndexes:
#             NucleusName, _ , _ = smallFuncs.NucleiSelection(ind )

#             inP  = subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz'
#             outP = subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz'
#             # crop = subject.Temp.address + '/CropMask.nii.gz'
#             outDebug = subject.Label.Temp.address + '/' + NucleusName + '_Cropped.nii.gz'

#             if os.path.isfile(outDebug) and params.preprocess.Debug.justForNow:
#                 copyfile(outDebug , outP)
#             else:
#                 os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )
#                 if params.preprocess.Debug.doDebug:
#                     copyfile(outP , outDebug)

def Bash_AugmentNonLinear(subject , subjectRef , outputAddress): # Image , Mask , Reference , output):

    print('     Augment NonLinear')

    ImageOrig = subject.address       + '/' + subject.ImageProcessed + '.nii.gz'
    # MaskOrig  = subject.Label.address + '/' + subject.Label.LabelProcessed + '.nii.gz'
    ImageRef  = subjectRef.address    + '/' + subjectRef.ImageProcessed + '.nii.gz'

    OutputImage = outputAddress  + '/' + subject.ImageProcessed + '.nii.gz'
    labelAdd    = smallFuncs.mkDir(outputAddress + '/Label')
    deformationAddr = smallFuncs.mkDir(outputAddress + '/Temp/deformation')


    if not os.path.isfile(OutputImage):
        os.system("ANTS 3 -m CC[%s, %s,1,5] -t SyN[0.25] -r Gauss[3,0] -o %s -i 30x90x20 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000"%(ImageOrig , ImageRef , deformationAddr + '/test') )
        os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(ImageOrig , OutputImage , ImageOrig , deformationAddr + '/testWarp.nii.gz') )

    _, _, names = smallFuncs.NucleiSelection(ind = 1)
    for name in names:
        MaskOrig  = subject.Label.address + '/' + name + '_PProcessed.nii.gz'
        OutputMask  = labelAdd + '/' + name + '_PProcessed.nii.gz'
        if not os.path.isfile(OutputMask):
            os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(MaskOrig , OutputMask , MaskOrig , deformationAddr + '/testWarp.nii.gz' ) )
