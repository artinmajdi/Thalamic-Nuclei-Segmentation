import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import otherFuncs.smallFuncs as smallFuncs
from shutil import copyfile
import nibabel as nib


def RigidRegistration(subject , Template , preprocess):

    
    processed = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.Temp.address + '/CropMask.nii.gz'
    LinearAffine = subject.Temp.Deformation.address + '/linearAffine.txt'
    if preprocess.Mode and preprocess.Cropping: # and not os.path.isfile(outP):
        print('     Rigid Registration')
        if not os.path.isfile(LinearAffine): 
            os.system("ANTS 3 -m CC[%s, %s ,1,5] -o %s -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(processed , Template.Image , subject.Temp.Deformation.address + '/linear') )

        if not os.path.isfile(outP): 
            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Mask , outP , processed , LinearAffine) )

def BiasCorrection(subject , params):
    
    inP  = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_bias_corr.nii.gz'
    if params.preprocess.Mode and params.preprocess.BiasCorrection:
        if os.path.isfile(outDebug):
            copyfile(outDebug , outP)
        else:
            print('     Bias Correction')            
            os.system( "N4BiasFieldCorrection -d 3 -i %s -o %s -b [200] -s 3 -c [50x50x30x20,1e-6]"%( inP, outP )  )
            if params.preprocess.save_debug_files:
                copyfile(outP , outDebug)

"""
def RigidRegistration_2AV(subject , Template , preprocess):
    
    
    processed = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    FullImage = subject.address + '/' + subject.ImageOriginal + '.nii.gz'
    
    
    LinearAffine_FullImage = subject.Temp.Deformation.address 
    LinearAffine_CropImage = subject.Temp.Deformation.address  + '_Crop'
    
    
    Template_CropImage = Template.Address + 'cropped_origtemplate.nii.gz'
    # Template_FullImage = Template.Address + 'origtemplate.nii.gz'
    
    outP_crop = subject.Temp.address + '/CropMask_AV.nii.gz'
    if preprocess.Cropping and not os.path.isfile(outP_crop):  
        
        if not os.path.isfile(LinearAffine_FullImage + '/linearAffine.txt' ): 
            print('     Rigid Registration of cropped Image')
            smallFuncs.mkDir(LinearAffine_CropImage)
            os.system("ANTS 3 -m CC[%s, %s ,1,5] -o %s -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(processed , Template_CropImage , LinearAffine_CropImage + '/linear') )
        
            print('     Warping of cropped Mask')
            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Address + 'CropMask_AV.nii.gz' , outP_crop , processed , LinearAffine_CropImage) )

        else:
            outP_full = subject.Temp.address + '/Mask_AV.nii.gz'
            mainCrop = subject.Temp.address + '/CropMask.nii.gz.nii.gz'

            print('     Warping of full Mask')
            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Address + 'Mask_AV.nii.gz' , outP_full , FullImage , LinearAffine_FullImage) )
            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Address + 'CropMaskV3.nii.gz' , outP_full , FullImage , LinearAffine_FullImage) )

            print('    Cropping the Full AV Mask')
            cropping_AV_Mask(outP_full, outP_crop, mainCrop)
"""

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

"""
def cropping_AV_Mask(inP, outP, crop):
    
    def cropImage_FromCoordinates(CropMask , Gap): 
        BBCord = smallFuncs.findBoundingBox(CropMask>0.5)

        d = np.zeros((3,2),dtype=np.int)
        for ix in range(len(BBCord)):
            d[ix,:] = [  BBCord[ix][0]-Gap[ix] , BBCord[ix][-1]+Gap[ix]  ]
            d[ix,:] = [  max(d[ix,0],0)    , min(d[ix,1],CropMask.shape[ix])  ]

        return d
            
    # crop = subject.Temp.address + '/CropMask.nii.gz' 
    
    d = cropImage_FromCoordinates(crop , [0,0,0])
    mskC = nib.load(inP).slicer[ d[0,0]:d[0,1], d[1,0]:d[1,1], d[2,0]:d[2,1] ]                    
    nib.save(mskC , outP)
"""