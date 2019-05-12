# import numpy as np
import os, sys
# sys.path.append(os.path.dirname(__file__))
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage
# import otherFuncs.smallFuncs as smallFuncs
# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# params = paramFunc.Run(UserInfo.__dict__, terminal=True)

class Input_cls():
    def __init__(self, dir_in='' , dir_out=''):

        def directories(self):
            for en in range(len(sys.argv)):

                if sys.argv[en].lower() in ('-i','--input'):    dir_in = sys.argv[en+1]
                elif sys.argv[en].lower() in ('-o','--output'): dir_out = sys.argv[en+1]
            self.dir_in, self.dir_out = dir_in , dir_out
                           
        directories(self)

        self.subjList = [s for s  in os.listdir(self.dir_in) if 'vimp' in s]

        
    def save_middleSlice_per_subject(self,subj):
        im = nib.load(self.dir_in + '/' + subj + '/WMnMPRAGE_bias_corr.nii.gz')
        imm = im.get_data()[...,int(im.shape[2]/2)]

        imm2 = skimage.transform.rotate(imm,90,resize=True)
        imageio.imwrite(self.dir_out + '/' + subj + '.jpg', imm2)  


input = Input_cls()

for subj in input.subjList:
    input.save_middleSlice_per_subject(subj)




# subj = 'vimp2_901_07052013_AS_MS'
# dir_in = '/home/artinl/Documents/RESEARCH/dataset/7T/' + subj + '/WMnMPRAGE_bias_corr.nii.gz'
# im = nib.load(dir_in)

# imm = im.get_data()[...,int(im.shape[2]/2)]

# plt.imshow(imm,cmap='gray')


# dir_out = '/home/artinl/Documents/RESEARCH/'

# imm2 = skimage.transform.rotate(imm,90,resize=True)
# imageio.imwrite(dir_out + subj + '.jpg', imm2)

