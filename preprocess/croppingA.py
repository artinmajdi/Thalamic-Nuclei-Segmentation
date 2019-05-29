import os
import sys
from shutil import copyfile

import nibabel as nib
import numpy as np
from skimage import measure

import otherFuncs.smallFuncs as smallFuncs

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main(subject , params):

    if params.preprocess.Mode and params.preprocess.Cropping.Mode:
        print('     Cropping')
        func_cropImage(params, subject)
    return True


def func_cropImage(params, subject):

    def cropImage_FromCoordinates(CropMask , Gap): 
        BBCord = smallFuncs.findBoundingBox(CropMask>0.5)

        d = np.zeros((3,2),dtype=np.int)
        for ix in range(len(BBCord)):
            d[ix,:] = [  BBCord[ix][0]-Gap[ix] , BBCord[ix][-1]+Gap[ix]  ]
            d[ix,:] = [  max(d[ix,0],0)    , min(d[ix,1],CropMask.shape[ix])  ]

        return d
            
    crop = subject.Temp.address + '/CropMask.nii.gz' 
    
    def check_crop(inP, outP, outDebug, CropCoordinates):

        def applyCropping(image):
            d = CropCoordinates
            return image.slicer[ d[0,0]:d[0,1], d[1,0]:d[1,1], d[2,0]:d[2,1] ]
                            
        if os.path.isfile(outDebug): 
            copyfile(outDebug , outP)
        else: 
            if 'ANTs' in params.preprocess.Cropping.Method:
                os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

            elif 'python' in params.preprocess.Cropping.Method:
                mskC = applyCropping( nib.load(inP))
                nib.save(mskC , outP)
                
            if params.preprocess.Debug.doDebug: copyfile(outP , outDebug)
                        
    def directoriesImage(subject):
        inP  = outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'                  
        outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_Cropped.nii.gz'
        return inP, outP, outDebug 

    def directoriesNuclei(subject, ind):
        NucleusName, _, _ = smallFuncs.NucleiSelection(ind )
        inP = outP = subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz'
        outDebug = subject.Label.Temp.address + '/' + NucleusName + '_Cropped.nii.gz'
        return inP, outP, outDebug

    inP, outP, outDebug = directoriesImage(subject)          
    CropCoordinates = cropImage_FromCoordinates(nib.load(crop).get_data() , [0,0,0])  if not os.path.isfile(outDebug) and 'python' in params.preprocess.Cropping.Method else ''

    check_crop(inP, outP, outDebug, CropCoordinates)

    for ind in params.WhichExperiment.Nucleus.FullIndexes:
        inP, outP, outDebug = directoriesNuclei(subject, ind)
        if not os.path.isfile(outDebug) and 'python' in params.preprocess.Cropping.Method and CropCoordinates == '': CropCoordinates = cropImage_FromCoordinates(nib.load(crop).get_data() , [0,0,0])  
        check_crop(inP, outP, outDebug, CropCoordinates)
