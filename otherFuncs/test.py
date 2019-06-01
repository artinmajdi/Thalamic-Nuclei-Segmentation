import nibabel as nib
import numpy as np
import nilearn
import matplotlib.pyplot as plt
import os, sys
from skimage import measure
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
params = paramFunc.Run(UserInfo.__dict__, terminal=False)


def cropImage_FromCoordinates(CropMask , Gap): 
    BBCord = smallFuncs.findBoundingBox(CropMask>0.5)

    d = np.zeros((3,2),dtype=np.int)
    for ix in range(len(BBCord)):
        d[ix,:] = [  BBCord[ix][0]-Gap[ix] , BBCord[ix][-1]+Gap[ix]  ]
        d[ix,:] = [  max(d[ix,0],0)    , min(d[ix,1],CropMask.shape[ix])  ]

    return d

mode = 'train'
Subjects = params.directories.Train.Input.Subjects if 'train' in mode else params.directories.Test.Input.Subjects
for _, subject in Subjects.items():
    
    # subject = Subjects[list(Subjects)[0]]    
    cropAV = nib.load(subject.Temp.address + '/CropMask_AV.nii.gz').get_data()
    mskAV  = nib.load(subject.Label.address + '/2-AV_PProcessed.nii.gz').get_data()

    
    if np.sum(cropAV) > 0:
        d = cropImage_FromCoordinates(cropAV , [0,0,0])  

        mskAV_Crp = nib.load(subject.Label.address + '/2-AV_PProcessed.nii.gz').slicer[ d[0,0]:d[0,1], d[1,0]:d[1,1], d[2,0]:d[2,1] ]            
        
        a = np.sum(mskAV_Crp.get_data()) / np.sum(mskAV)
        flag = 'Correct' if np.abs(1-a) < 0.001 else 'Clipped ' + str(a)
        print(subject.subjectName  , '------- <' , flag , '>---')
    else:
        print(subject.subjectName  , 'zero mask')
        # B = mskAV*(1-cropAV>0.5)
        # print(np.unique(B))