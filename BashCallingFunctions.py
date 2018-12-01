import os
import numpy as np
from smallFuncs import mkDir
from shutil import copyfile

def RigidRegistration(subject , Template , preprocess):

#     processed = subject.Address + '/' + subject.origImage + '_bias_corr.nii.gz'
    processed = subject.Address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.Temp.Address + '/CropMask.nii.gz'

    if preprocess.Mode and preprocess.Cropping.Mode:
        os.system("ANTS 3 -m CC[%s, %s ,1,5] -o linear -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(processed , Template.Image) )
        os.system("WarpImageMultiTransform 3 %s %s -R %s linearAffine.txt"%(Template.Mask , outP , processed) )
        
def BiasCorrection(subject , preprocess):

    inP  = subject.Address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.Address + '/' + subject.ImageProcessed + '.nii.gz'
    
    if preprocess.Mode and preprocess.BiasCorrection.Mode:
        os.system("N4BiasFieldCorrection -d 3 -i %s -o %s -b [200] -s 3 -c [50x50x30x20,1e-6]"%( inP, outP )  )

        if preprocess.Debug.Mode:
            copyfile(outP , subject.Temp.Address + '/' + subject.ImageOriginal + '_bias_corr.nii.gz')

def Bash_Cropping(subject , preprocess):

    if preprocess.Mode and preprocess.Cropping.Mode:

        inP  = subject.Address + '/' + subject.ImageProcessed + '.nii.gz'
        outP = subject.Address + '/' + subject.ImageProcessed + '.nii.gz'
        crop = subject.Temp.Address + '/CropMask.nii.gz'
        os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

        if preprocess.Debug.Mode:
            copyfile(outP , subject.Temp.Address + '/' + subject.ImageOriginal + '_bias_corr_Cropped.nii.gz')

        # Cropping the Label
        inP  = subject.Label.Address + '/' + subject.Label.LabelProcessed + '.nii.gz'
        outP = subject.Label.Address + '/' + subject.Label.LabelProcessed + '.nii.gz'
        crop = subject.Temp.Address + '/CropMask.nii.gz'
        os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

        if preprocess.Debug.Mode:           
            copyfile(outP , subject.Label.Temp.Address + '/' + subject.Label.LabelOriginal + '_Cropped.nii.gz')        

def Bash_AugmentNonLinear(subject , subjectRef , outputAddress): # Image , Mask , Reference , output):

    ImageOrig = subject.Address       + '/' + subject.ImageProcessed + '.nii.gz' 
    MaskOrig  = subject.Label.Address + '/' + subject.Label.LabelProcessed + '.nii.gz'   
    ImageRef  = subjectRef.Address    + '/' + subjectRef.ImageProcessed + '.nii.gz' 

    OutputImage = outputAddress  + '/' + subject.ImageProcessed + '.nii.gz' 
    labelAdd = mkDir(outputAddress + '/Label') 
    OutputMask  = labelAdd + '/' + subject.Label.LabelProcessed + '.nii.gz' 
    deformationAddr = mkDir(outputAddress + '/Temp/deformation')           
    

    if not os.path.isfile(OutputImage):
        os.system("ANTS 3 -m CC[%s, %s,1,5] -t SyN[0.25] -r Gauss[3,0] -o %s -i 30x90x20 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000"%(ImageOrig , ImageRef , deformationAddr + '/test') )
        os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(ImageOrig , OutputImage , ImageOrig , deformationAddr + '/testWarp.nii.gz') )

    if not os.path.isfile(OutputMask):
        os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(MaskOrig , OutputMask , MaskOrig , deformationAddr + '/testWarp.nii.gz' ) )