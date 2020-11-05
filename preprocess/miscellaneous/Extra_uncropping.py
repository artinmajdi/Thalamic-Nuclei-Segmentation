import os, sys
import nibabel as nib
import numpy as np
from shutil import copyfile
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from otherFuncs import smallFuncs
from preprocess import uncrop
                        


class UserEntry:
    def __init__(self):
        self.dir_in  = ''
        self.dir_out = ''
        self.dir_mask = ''
        self.mode    = 0

        for en in range(len(sys.argv)):
            if sys.argv[en].lower() in ('-i','--input'):    self.dir_in  = os.path.abspath(sys.argv[en+1])
            elif sys.argv[en].lower() in ('-o','--output'): self.dir_out = os.path.abspath(sys.argv[en+1]) 
            elif sys.argv[en].lower() in ('-msk','--mask'): self.dir_mask = os.path.abspath(sys.argv[en+1]) 
            elif sys.argv[en].lower() in ('-m','--_mode'):   self.mode    = sys.argv[en+1]


            
class uncrop_cls:
    def __init__(self, dir_in = '' , dir_out = '' , dir_mask = '' , maskCrop=''):

        self.dir_in  = dir_in
        self.dir_out = dir_out
        self.dir_mask = dir_mask
        self.maskCrop = maskCrop

    def apply_uncrop(self):

        smallFuncs.mkDir(self.dir_out + '/Label')   
 
        
        for label in smallFuncs.Nuclei_Class().All_Nuclei().Names:
            input_image  = self.dir_in  + '/Label/' + label    + '.nii.gz'
            output_image = self.dir_out + '/Label/' + label    + '.nii.gz'
            full_mask = self.dir_in  + '/Label/mask_inp.nii.gz' 

            uncrop.uncrop_by_mask(input_image=input_image, output_image=output_image , full_mask=full_mask)     

    def uncrop_All(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'cas' in s]:
            print(subj , '\n')
            dir_in  = self.dir_in + '/' + subj
            dir_out = self.dir_out + '/' + subj
            temp = uncrop_cls(dir_in=dir_in , dir_out=dir_out , maskCrop=self.maskCrop)
            temp.apply_uncrop()

    def apply_single_file(self):
        uncrop.uncrop_by_mask(input_image=self.dir_in, output_image=self.dir_out , full_mask=self.dir_mask) 



UI = UserEntry()


if UI.mode == '0': 
    uncrop_cls(dir_in = UI.dir_in , dir_out = UI.dir_out, dir_mask = '' , maskCrop='mask_inp').apply_uncrop()
elif UI.mode == 'all':            
    uncrop_cls(dir_in = UI.dir_in , dir_out = UI.dir_out, dir_mask = '' , maskCrop='mask_inp').uncrop_All()
elif UI.mode == 'single':
    uncrop_cls(dir_in = UI.dir_in , dir_out = UI.dir_out, dir_mask = UI.dir_mask , maskCrop='').apply_single_file()