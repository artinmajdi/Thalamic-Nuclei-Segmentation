import os
import sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras') # sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import otherFuncs.smallFuncs as smallFuncs
from shutil import copyfile
from skimage.measure import regionprops, label
import Parameters.UserInfo as UserInfo
from Parameters import paramFunc
# params = paramFunc.Run(UserInfo.__dict__, terminal=False)
import nibabel as nib

def main(dir_in, dir_template):
    class UserEntry():
        def __init__(self, dir_in  = '' , dir_template='/array/ssd/msmajdi/code/thalamus/keras/general/RigidRegistration', mode=0):
            self.dir_in       = dir_in
            self.dir_template = dir_template
            self.mode = mode

            for en in range(len(sys.argv)):
                if sys.argv[en].lower() in ('-i','--input'):      self.dir_in       = os.getcwd() + '/' + sys.argv[en+1]
                elif sys.argv[en].lower() in ('-t','--template'): self.dir_template = sys.argv[en+1]
                elif sys.argv[en].lower() in ('-m','--mode'):     self.mode         = int(sys.argv[en+1])                     

    def mkDir(Dir):
        if not os.path.isdir(Dir): os.makedirs(Dir)
        return Dir

    class AV_crop_cls():
        def __init__(self, dir_in = '' , dir_template=''):

            self.dir_in       = dir_in
            self.dir_template = dir_template

        def apply_Warp(self):
                    
            inMask  = self.dir_template + '/CropMask_AV.nii.gz'
            Image   = self.dir_in + '/PProcessed.nii.gz' 
            outMask = self.dir_in + '/temp/CropMask_AV.nii.gz'  
            LinearAffine = self.dir_in + '/temp/deformation/linearAffine.txt'

            if os.path.isfile(LinearAffine): #  and not os.path.isfile(outMask): 
                os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(inMask , outMask , Image , LinearAffine) )
            elif not os.path.isfile(LinearAffine): 
                print('Registration is required', self.dir_in)


            # subject.address = self.dir_in
            # subject.ImageProcessed = 'PProcessed'
            # subject.Temp.address = self.dir_in + '/temp'
            # subject.Temp.Deformation.address = self.dir_in + '/temp/deformation'

            # BashCallingFunctionsA.RigidRegistration( subject , params.WhichExperiment.HardParams.Template , params.preprocess)
            
            # croppingA.main(subject , params)

            # Images = [s for s in os.listdir(self.dir_in) if 'PProcessed' not in s and '.nii.gz' in s]
            # FullImage = self.dir_in + '/' + Images[0]
            # outP = mkDir(self.dir_in + '/temp/') + 'CropMask.nii.gz'
            # os.system("ANTS 3 -m CC[%s, %s ,1,5] -o %s -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(FullImage , self.dir_template + '/origtemplate.nii.gz', mkDir(self.dir_in + 'temp/deformation/') + 'linear') )

            # os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(self.dir_template + '/CropMask.nii.gz' , outP , FullImage , LinearAffine) )
            # os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(inMask , outMask , Image , LinearAffine) )

        def warp_all(self):
            for subj in [s for s in os.listdir(self.dir_in) if 'vimp' in s]:
                print(subj , '\n')
                dir_in       = self.dir_in + '/' + subj
                dir_template = self.dir_template 

                temp = AV_crop_cls(dir_in = dir_in , dir_template=dir_template)
                temp.apply_Warp()


    UI = UserEntry(dir_in=dir_in, dir_template=dir_template)
    # UI.dir_in  = '/array/ssd/msmajdi/experiments/keras/exp5_CSFn/train/Main/vimp2_845_05312013_VZ'
    # UI.mode = 0

    if UI.mode == 0: AV_crop_cls(dir_in = UI.dir_in , dir_template = UI.dir_template).apply_Warp()
    else:            AV_crop_cls(dir_in = UI.dir_in , dir_template = UI.dir_template).warp_all()

def save_Crop_AV():

    dir_template='/array/ssd/msmajdi/code/thalamus/keras/general/RigidRegistration'
    dir = dir_template + '/AV_Masks/'

    subjects = [s for s in os.listdir(dir) if 'vimp2' in s]

    bbox_list = np.zeros((len(subjects),6))
    for ix, subj in enumerate(subjects):
        msk = nib.load(dir + subj).get_data()

        obj = regionprops(label(msk))
        # print(obj[0].bbox , subj)
        bbox_list[ix,:] = obj[0].bbox

    D = np.append(bbox_list[:,:3].min(axis=0) , bbox_list[:,3:].max(axis=0) )
    D = [int(i) for i in D]

    im = nib.load(dir_template + '/cropped_origtemplate.nii.gz')
    msk1 = np.zeros(im.shape)

    g = 3
    msk1[D[0]-g:D[3]+g,D[1]-g:D[4]+g,D[2]-g:D[5]+g] = 1

    smallFuncs.saveImage(msk1,im.affine, im.header , dir_template+'/CropMask_AV.nii.gz')

    os.system("uncrop -i CropMask_AV.nii.gz -o Mask_AV2.nii.gz -msk CropMaskV3.nii.gz -m 2")

# save_Crop_AV()

"""
def check_if_AV_inside_Crop():

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
"""

def check_if_AV_inside_Crop():

    class UserEntry():
        def __init__(self, dir_in  = '' , mode=1):
            self.dir_in       = dir_in
            self.mode = mode

            for en in range(len(sys.argv)):
                if sys.argv[en].lower() in ('-i','--input'):  self.dir_in = os.getcwd() + '/' + sys.argv[en+1]
                elif sys.argv[en].lower() in ('-m','--mode'): self.mode   = int(sys.argv[en+1])


    def cropImage_FromCoordinates(CropMask , Gap): 
        BBCord = smallFuncs.findBoundingBox(CropMask>0.5)

        d = np.zeros((3,2),dtype=np.int)
        for ix in range(len(BBCord)):
            d[ix,:] = [  BBCord[ix][0]-Gap[ix] , BBCord[ix][-1]+Gap[ix]  ]
            d[ix,:] = [  max(d[ix,0],0)    , min(d[ix,1],CropMask.shape[ix])  ]

        return d

    dir = '/array/ssd/msmajdi/experiments/keras/exp6/crossVal/ET/d/'
    for subject in [s for s in os.listdir(dir) if 'vimp' in s]:
        
        # subject = Subjects[list(Subjects)[0]]    
        cropAV = nib.load(dir + subject + '/temp/CropMask_AV.nii.gz').get_data()
        mskAV  = nib.load(dir + subject + '/Label/2-AV_PProcessed.nii.gz').get_data()
        
        if np.sum(cropAV) > 0:
            d = cropImage_FromCoordinates(cropAV , [0,0,0])  

            mskAV_Crp = nib.load(dir + subject + '/Label/2-AV_PProcessed.nii.gz').slicer[ d[0,0]:d[0,1], d[1,0]:d[1,1], d[2,0]:d[2,1] ]            
            
            a = np.sum(mskAV_Crp.get_data()) / np.sum(mskAV)
            flag = 'Correct' if np.abs(1-a) < 0.001 else 'Clipped ' + str(a)
            print(subject  , '------- <' , flag , '>---')
        else:
            print(subject  , 'zero mask')
            # B = mskAV*(1-cropAV>0.5)
            # print(np.unique(B))

# check_if_AV_inside_Crop()
