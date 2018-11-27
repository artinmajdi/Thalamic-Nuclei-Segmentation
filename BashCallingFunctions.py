import os
import numpy as np
from smallFuncs import mkDir

def RigidRegistration(subject , Template):

    BiasCorrected = subject.Address + '/' + subject.origImage + '_bias_corr.nii.gz'
    outP = subject.Address + '/CropMask.nii.gz'

    if ( not os.path.isfile( outP ) )  and ( '_bias_corr_Cropped' not in subject.Cropped ):
        os.system("ANTS 3 -m CC[%s, %s ,1,5] -o linear -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(BiasCorrected , Template.Image) )
        os.system("WarpImageMultiTransform 3 %s %s -R %s linearAffine.txt"%(Template.Mask , outP , BiasCorrected) )

def BiasCorrection(subject):

    inP  = subject.Address + '/' + subject.origImage + '.nii.gz'
    outP = subject.Address + '/' + subject.origImage + '_bias_corr.nii.gz'
    
    if ( not os.path.isfile(outP) ) and ( '_bias_corr_Cropped' not in subject.Cropped ):
        os.system("N4BiasFieldCorrection -d 3 -i %s -o %s -b [200] -s 3 -c [50x50x30x20,1e-6]"%( inP, outP )  )

def Bash_Cropping(subject):

    inP  = subject.Address + '/' + subject.origImage + '_bias_corr.nii.gz'
    outP = subject.Address + '/' + subject.origImage + '_bias_corr_Cropped.nii.gz'
    crop = subject.Address + '/' + 'CropMask.nii.gz'

    if ( not os.path.isfile( outP ) ) and ( '_bias_corr_Cropped' not in subject.Cropped ):
        os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

    # Cropping the Label
    inP  = subject.Nucleus.Address + '/' + subject.Nucleus.Full + '.nii.gz'
    outP = subject.Nucleus.Address + '/' + subject.Nucleus.Full + '_Cropped.nii.gz'
    crop = subject.Address + '/' + 'CropMask.nii.gz'

    if ( not os.path.isfile( outP ) ) and ( '_bias_corr_Cropped' not in subject.Cropped ):
        os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

def Bash_AugmentNonLinear(Image , Mask , Reference , output):

    OutputImage = output.Address + '/' + output.Image + '.nii.gz' 
    OutputMask  = output.Address + '/Manual_Delineation_Sanitized/' + output.Mask + '.nii.gz' 
    mkDir(output.Address + '/deformation')
    mkDir(output.Address + '/Manual_Delineation_Sanitized')            
    

    if not os.path.isfile(OutputImage):
        os.system("ANTS 3 -m CC[%s, %s,1,5] -t SyN[0.25] -r Gauss[3,0] -o %s -i 30x90x20 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000"%(Image , Reference , output.Address + '/deformation/test') )
        os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(Image , OutputImage , Image , output.Address + '/deformation/testWarp.nii.gz') )

    if not os.path.isfile(OutputMask):
        os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(Mask , OutputMask , Mask , output.Address + '/deformation/testWarp.nii.gz' ) )