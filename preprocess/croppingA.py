import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from shutil import copyfile
from otherFuncs import smallFuncs
import nibabel as nib


def main(subject , params):

    def applyCropping(image, coordinates):
        d = coordinates
        return image.slicer[ d[0,0]:d[0,1], d[1,0]:d[1,1], d[2,0]:d[2,1] ]

    def check_cropImage(subject):
        
        def directoriesImage(subject):
            inP  = outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
            crop = subject.Temp.address + '/CropMask.nii.gz'       
            outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_Cropped.nii.gz'

            return inP, outP, crop, outDebug 
                
        def main_CropImage(inP, outP, crop, outDebug):

            CropCoordinates = ''
            if 'ANTs' in params.preprocess.Cropping.Method:
                os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )
                
            elif 'python' in params.preprocess.Cropping.Method:
                Gap = [0,0,1]
                im = nib.load(inP)
                CropMask = nib.load(crop)
                CropCoordinates = cropImage_FromCoordinates(CropMask.get_data() , Gap)

                imC = applyCropping(im, CropCoordinates)
                nib.save(imC , outP)
                
            if params.preprocess.Debug.doDebug: copyfile(outP , outDebug)

            return CropCoordinates

        CropCoordinates = ''
        inP, outP, crop, outDebug = directoriesImage(subject)

        if os.path.isfile(outDebug): copyfile(outDebug , outP)
        else: CropCoordinates = main_CropImage(inP, outP, crop, outDebug)
        
        return CropCoordinates
           
    def loopOver_Nuclei(params, subject , CropCoordinates ):

        def check_cropNucleus(subject, CropCoordinates, ind):
            
            def main_cropNucleus(params, subject, inP, outP, crop, CropCoordinates, outDebug):
                    
                if 'ANTs' in params.preprocess.Cropping.Method:
                    cropFailure = os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

                    # if cropFailure != 0:    
                    #     NucleusName, _, _ = smallFuncs.NucleiSelection(ind , params.WhichExperiment.Nucleus.Organ)
                    #     print(' cropping failed - >  copying image header into designated label',NucleusName)                    
                    #     os.system("CopyImageHeaderInformation %s %s %s"%(subject.address + '/' + subject.ImageProcessed + '.nii.gz', inP, inP))
                    #     cropFailure = os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

                elif 'python' in params.preprocess.Cropping.Method:
                    mskC = applyCropping( nib.load(inP) , CropCoordinates)
                    nib.save(mskC , outP)

                if params.preprocess.Debug.doDebug:
                    copyfile(outP , outDebug)
                    
            def directoriesNuclei(subject, ind):
                NucleusName, _, _ = smallFuncs.NucleiSelection(ind , params.WhichExperiment.Nucleus.Organ)
                inP = outP = subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz'
                crop = subject.Temp.address + '/CropMask.nii.gz'
                outDebug = subject.Label.Temp.address + '/' + NucleusName + '_Cropped.nii.gz'

                return inP, outP, crop, outDebug
                    
            inP, outP, crop, outDebug = directoriesNuclei(subject, ind)

            if os.path.isfile(outDebug): copyfile(outDebug , outP)
            else: main_cropNucleus(params, subject, inP, outP, crop, CropCoordinates, outDebug)

        for ind in params.WhichExperiment.Nucleus.FullIndexes:
            check_cropNucleus(subject, CropCoordinates, ind)

    if params.preprocess.Mode and params.preprocess.Cropping.Mode:
        print('     Cropping')

        if 'python' in params.preprocess.Cropping.Method:
            CropCoordinates = check_cropImage(subject)
            loopOver_Nuclei(params, subject , CropCoordinates )
        else:
            loopOver_Nuclei(params, subject , '' )
            check_cropImage(subject)


    return True


   
def cropImage_FromCoordinates(CropMask , Gap): # func_CropCoordinates
    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 10)[0]
    c2 = np.where(np.sum(ss,axis=0) > 10)[0]

    ss = np.sum(CropMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 10)[0]

    BBCord = [   [c1[0],c1[-1]]  ,  [c2[0],c2[-1]]  , [c3[0],c3[-1]]  ]


    d = np.zeros((3,2),dtype=np.int)
    for ix in range(len(BBCord)):
        d[ix,:] = [  BBCord[ix][0]-Gap[ix] , BBCord[ix][-1]+Gap[ix]  ]
        d[ix,:] = [  max(d[ix,0],0)    , min(d[ix,1],CropMask.shape[ix])  ]

    return d



