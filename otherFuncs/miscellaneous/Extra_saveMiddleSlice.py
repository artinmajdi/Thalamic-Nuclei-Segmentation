# import numpy as np
import os, sys

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage


class Input_cls():
    def __init__(self, dir_in='' , dir_out=''):
        self.dir_in = dir_in
        self.dir_out = dir_out
        def directories(self):
            for en in range(len(sys.argv)):

                if sys.argv[en].lower() in ('-i','--input'):    dir_in = sys.argv[en+1]
                elif sys.argv[en].lower() in ('-o','--output'): dir_out = sys.argv[en+1]
            self.dir_in, self.dir_out = dir_in , dir_out
                           
        directories(self)

        self.subjList = [s for s  in os.listdir(self.dir_in) if ('case' in s) and ('jpg' not in s)]

        
    def middleSlice(self):
        msk = nib.load(self.dir_in + '/' + self.subj + '/Label/1-THALAMUS.nii.gz')
        objects = skimage.measure.regionprops(skimage.measure.label(msk.get_fdata()))
        Ix = np.argsort( [obj.area for obj in objects] )
        bbox = objects[ Ix[-1] ].bbox
        return int((bbox[2] + bbox[5])/2) 
        
    def save_image(self , imm):
        imm2 = skimage.transform.rotate(imm,90,resize=True)
        imageio.imwrite(self.dir_out + '/' + self.subj + '.jpg', imm2)  

    def save_subject_middle_jpg(self,subj):
        
        self.subj = subj
        m = self.middleSlice()

        im = nib.load(self.dir_in + '/' + self.subj + '/WMnMPRAGE_bias_corr.nii.gz')
        imm = im.slicer[:,:,m:m+1].get_fdata()
        
        self.save_image(imm)



input = Input_cls()

for subj in input.subjList:
    input.save_subject_middle_jpg(subj)




# subj = 'vimp2_901_07052013_AS_MS'
# Imdir_in = '/home/artinl/Documents/RESEARCH/dataset/7T/' + subj + '/WMnMPRAGE_bias_corr.nii.gz'
# Mskdir_in = '/home/artinl/Documents/RESEARCH/dataset/7T/' + subj + '/Label/1-THALAMUS.nii.gz'
# msk = nib.load(Mskdir_in)


# a = msk.slicer[:,:,100:101]
# a.get_fdata()
# b = a[:]
# np.array(a)
# objects = skimage.measure.regionprops(skimage.measure.label(msk.get_fdata()))

# Ix = np.argsort( [obj.area for obj in objects] )
# bbox = objects[ Ix[-1] ].bbox

# im = nib.load(Imdir_in)
# imm = im.get_fdata()[...,int((bbox[2] + bbox[5])/2) ]

# plt.imshow(imm,cmap='gray')


# dir_out = '/home/artinl/Documents/RESEARCH/'

# imm2 = skimage.transform.rotate(imm,90,resize=True)
# imageio.imwrite(dir_out + subj + '.jpg', imm2)

