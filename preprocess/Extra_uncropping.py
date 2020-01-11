import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
from preprocess import uncrop
# from nilearn import image as niImage
import nibabel as nib
import numpy as np
from shutil import copyfile                              


class UserEntry():
    def __init__(self):
        self.dir_in  = ''
        self.dir_out = ''
        self.dir_mask = ''
        self.mode    = 0

        for en in range(len(sys.argv)):
            if sys.argv[en].lower() in ('-i','--input'):    self.dir_in  = os.getcwd() + '/' + sys.argv[en+1] if '/array/ssd' not in sys.argv[en+1] else sys.argv[en+1] 
            elif sys.argv[en].lower() in ('-o','--output'): self.dir_out = os.getcwd() + '/' + sys.argv[en+1] if '/array/ssd' not in sys.argv[en+1] else sys.argv[en+1] 
            elif sys.argv[en].lower() in ('-msk','--mask'): self.dir_mask = os.getcwd() + '/' + sys.argv[en+1] if '/array/ssd' not in sys.argv[en+1] else sys.argv[en+1] 
            elif sys.argv[en].lower() in ('-m','--mode'):   self.mode    = sys.argv[en+1]
            
class uncrop_cls():
    def __init__(self, dir_in = '' , dir_out = '' , dir_mask = '' , maskCrop=''):

        self.dir_in  = dir_in
        self.dir_out = dir_out
        self.dir_mask = dir_mask
        self.maskCrop = maskCrop

    def apply_uncrop(self):

        smallFuncs.mkDir(self.dir_out + '/Label')   

        # image = [n for n in os.listdir(self.dir_in) if '.nii.gz' in n]
        # copyfile(self.dir_in + '/' + image[0] , self.dir_out + '/' + image[0])

        # input_image  = self.dir_in  + '/crop_t1.nii.gz'
        # output_image = self.dir_out + '/crop_t1.nii.gz'
        # full_mask    = self.dir_in  + '/Label/' + self.maskCrop + '.nii.gz'
        # uncrop.uncrop_by_mask(input_image=input_image, output_image=output_image , full_mask=full_mask)  
        
        for label in smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Names:
            input_image  = self.dir_in  + '/Label/' + label    + '_PProcessed.nii.gz'
            output_image = self.dir_out + '/Label/' + label    + '.nii.gz'
            # full_mask = self.dir_in  + '/Label/' + self.maskCrop + '.nii.gz' 
            full_mask = self.dir_in  + '/temp/CropMask.nii.gz' 

            uncrop.uncrop_by_mask(input_image=input_image, output_image=output_image , full_mask=full_mask)     

    def uncrop_All(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'vimp' in s]:
            print(subj , '\n')
            dir_in  = self.dir_in + '/' + subj
            dir_out = self.dir_out + '/' + subj
            temp = uncrop_cls(dir_in=dir_in , dir_out=dir_out , maskCrop=self.maskCrop)
            temp.apply_uncrop()

    def apply_individual(self):
        uncrop.uncrop_by_mask(input_image=self.dir_in, output_image=self.dir_out , full_mask=self.dir_mask) 



UI = UserEntry()
# UI.dir_in  = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset2_with_Manual_Labels/full_Image/freesurfer/ManualLabels2'
# UI.dir_out = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset2_with_Manual_Labels/full_Image/freesurfer/ManualLabels2_uncropped'
# UI.mode    = 1
if UI.mode == 0: 
    uncrop_cls(dir_in = UI.dir_in , dir_out = UI.dir_out, dir_mask = '' , maskCrop='mask_inp').apply_uncrop()
elif UI.mode == 'all':            
    uncrop_cls(dir_in = UI.dir_in , dir_out = UI.dir_out, dir_mask = '' , maskCrop='mask_inp').uncrop_All()
elif UI.mode == 'single':
    uncrop_cls(dir_in = UI.dir_in , dir_out = UI.dir_out, dir_mask = UI.dir_mask , maskCrop='').apply_individual()