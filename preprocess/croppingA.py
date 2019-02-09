import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from shutil import copyfile
from otherFuncs import smallFuncs
import nibabel as nib


def main(subject , params):

    if params.preprocess.Cropping.Mode:

        inP  = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
        outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
        crop = subject.Temp.address + '/CropMask.nii.gz'
        outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_Cropped.nii.gz'

        if os.path.isfile(outDebug) and params.preprocess.Debug.justForNow:
            copyfile(outDebug , outP)
        else:

            if 'ANTs' in params.preprocess.Cropping.Method:
                os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )
            elif 'python' in params.preprocess.Cropping.Method:
                Gap = [0,0,1]
                im = nib.load(inP)
                CropMask = nib.load(crop)
                imC , CropCoordinates = funcCropping_Mode(im.get_data() , CropMask.get_data() , Gap)
                smallFuncs.saveImage(imC , im.affine , im.header , outP)

            if params.preprocess.Debug.doDebug:
                copyfile(outP , outDebug)

        # Cropping the Label
        for ind in params.WhichExperiment.Nucleus.FullIndexes:
            NucleusName, _ = smallFuncs.NucleiSelection(ind , params.WhichExperiment.Nucleus.Organ)

            inP  = subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz'
            outP = subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz'
            # crop = subject.Temp.address + '/CropMask.nii.gz'
            outDebug = subject.Label.Temp.address + '/' + NucleusName + '_Cropped.nii.gz'

            if os.path.isfile(outDebug) and params.preprocess.Debug.justForNow:
                copyfile(outDebug , outP)
            else:

                if 'ANTs' in params.preprocess.Cropping.Method:
                    os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )
                elif 'python' in params.preprocess.Cropping.Method:
                    msk = nib.load(inP)
                    mskC = cropFromCoordinates(msk.get_data(), CropCoordinates)
                    smallFuncs.saveImage(mskC , msk.affine , msk.header , outP)

                if params.preprocess.Debug.doDebug:
                    copyfile(outP , outDebug)


    return True

# def cropFunc(im , cc , Gap):

#     szOg = im.shape

#     d = np.zeros((3,2))
#     for ix in range(len(cc)):
#         d[ix,:] = [  cc[ix][0]-Gap[ix] , cc[ix][ cc[ix].shape[0]-1 ]+Gap[ix]  ]
#         d[ix,:] = [  max(d[ix,0],0)    , min(d[ix,1],szOg[ix])  ]

#     # d1 = [  c1[0]-Gap[0] , c1[ c1.shape[0]-1 ]+Gap[0]  ]
#     # d2 = [  c2[0]-Gap[1] , c2[ c2.shape[0]-1 ]+Gap[1]  ]
#     # SN = [  c3[0]-Gap[2] , c3[ c3.shape[0]-1 ]+Gap[2]  ]

#     # d1 = [max(d1[0],0) , min(d1[1],szOg[0])]
#     # d2 = [max(d2[0],0) , min(d2[1],szOg[0])]
#     # SN = [max(SN[0],0) , min(SN[1],szOg[0])]

#     d1 = d[0]
#     d1 = d[1]
#     SN = d[2]

#     SliceNumbers = range(SN[0],SN[1])

#     im = im[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers ]
#     CropCoordinates = [d1,d2,SliceNumbers]

#     return im , CropCoordinates

def cropFromCoordinates(im, CropCoordinates):
    d1 = CropCoordinates[0]
    d2 = CropCoordinates[1]
    SN = CropCoordinates[2]
    # SliceNumbers = CropCoordinates[2] 

    return im[   d1[0]:d1[1]  ,  d2[0]:d2[1]  ,  SN[0]:SN[1]   ]

def func_CropCoordinates(CropMask):
    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 10)[0]
    c2 = np.where(np.sum(ss,axis=0) > 10)[0]

    ss = np.sum(CropMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 10)[0]

    BBCord = [   [c1[0],c1[-1]]  ,  [c2[0],c2[-1]]  , [c3[0],c3[-1]]  ]

    return BBCord

# def funcCropping_Mode(im , CropMask , Gap):
    
#     c1,c2,c3 = func_CropCoordinates(CropMask)
#     im , CropCoordinates = cropFunc(im , [c1,c2,c3] , Gap)

#     return im , CropCoordinates

