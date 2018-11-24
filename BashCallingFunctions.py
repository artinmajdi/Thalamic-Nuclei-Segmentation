import os

def RigidRegistration(Input , Template):
    
    BiasCorrected = Input.Address + '/' + Input.Files.origImage + '_bias_corr.nii.gz' 
    outP = Input.Address + '/CropMask.nii.gz' 

    if not os.path.isfile( outP ):
        os.system("ANTS 3 -m CC[%s, %s ,1,5] -o linear -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(BiasCorrected , Template.Image) )
        os.system("WarpImageMultiTransform 3 %s %s -R %s linearAffine.txt"%(Template.Mask , outP , BiasCorrected) )

def BiasCorrection(Input):

    inP  = Input.Address + '/' + Input.Files.origImage + '.nii.gz' 
    outP = Input.Address + '/' + Input.Files.origImage + '_bias_corr.nii.gz' 

    if not os.path.isfile( outP ):
        os.system("N4BiasFieldCorrection -d 3 -i %s -o %s -b [200] -s 3 -c [50x50x30x20,1e-6]"%( inP, outP )  )

def Cropping(Input):

    inP  = Input.Address + '/' + Input.Files.origImage + '_bias_corr.nii.gz'
    outP = Input.Address + '/' + Input.Files.origImage + '_bias_corr_Cropped.nii.gz'
    crop = Input.Address + '/' + Input.Files.CropMask + '.nii.gz'
    
    if not os.path.isfile( outP ):    
        os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

    # Cropping the Label
    inP  = Input.Files.Nucleus.Address + '/' + Input.Files.Nucleus.Full + '.nii.gz'
    outP = Input.Files.Nucleus.Address + '/' + Input.Files.Nucleus.Full + '_Cropped.nii.gz'
    crop = Input.Address + '/' + Input.Files.CropMask + '.nii.gz'
    
    if not os.path.isfile( outP ):    
        os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )        


