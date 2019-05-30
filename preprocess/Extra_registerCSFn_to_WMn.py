import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
from preprocess import uncrop
from nilearn import image as niImage
import nibabel as nib
import numpy as np
from shutil import copyfile                              


class UserEntry():
    def __init__(self):
        self.dir_in  = ''
        self.dir_out = ''
        self.mode    = 0

        for en in range(len(sys.argv)):
            if sys.argv[en].lower() in ('-i','--input'):    self.dir_in  = sys.argv[en+1]
            elif sys.argv[en].lower() in ('-o','--output'): self.dir_out = sys.argv[en+1]
            elif sys.argv[en].lower() in ('-m','--mode'):   self.mode    = int(sys.argv[en+1])
            
class register_cls():
    def __init__(self, dir_in = '' , dir_out = '' , maskCrop='mask_inp'):

        self.dir_in  = dir_in
        self.dir_out = dir_out
        self.maskCrop = maskCrop

    def apply_register(self):
        
        IN  = self.dir_in  
        OUT = self.dir_out 
        
        os.system('cd %s | antsRegistration  -d 3 --float 0 --output \[aff,affine.nii.gz\] -r \[crop_t1.nii.gz, crop_wmn.nii.gz,1\] -t Rigid\[0.1\] --metric MI\[crop_t1.nii.gz, crop_wmn.nii.gz,1,32,Regular,0.25\] --convergence \[1000x500x250x100,1e-7,10\] -v -f 8x4x2x1 -s 3x2x1x0vox -v -t Affine\[0.1\] --metric MI\[crop_t1.nii.gz, crop_wmn.nii.gz,1,32,Regular,0.25\] --convergence \[1000x500x250x100,5e-8,10\] -f 8x4x2x1 -s 3x2x1x0vox' % self.dir_in)
        
        smallFuncs.mkDir(self.dir_out + '/Label') 
        for nucleus in smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Names:
            IN  = self.dir_in  + '/Label/' + nucleus    + '.nii.gz'
            OUT = self.dir_out + '/Label/' + nucleus    + '.nii.gz'            
            os.system('cd %s | antsApplyTransforms -d 3 -i Label/%s.nii.gz -r crop_t1.nii.gz -o %s/Label/%s.nii.gz -t aff0GenericAffine.mat -n NearestNeighbor' % (IN , nucleus , OUT , nucleus))  

    def register_All(self):
        for subj in [s for s in os.listdir(self.dir_in)]:
            print(subj , '\n')
            dir_in  = self.dir_in + '/' + subj
            dir_out = self.dir_out + '/' + subj
            temp = register_cls(dir_in=dir_in , dir_out=dir_out)
            temp.apply_register()




# dir_in  = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn_Orig/WMn/case1'
# dir_out = smallFuncs.mkDir('/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/WMn/case1')

UI = UserEntry()
# UI.dir_in  = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn_Orig/CSFn/case1'
# UI.dir_out = smallFuncs.mkDir('/array/ssd/msmajdi/data/preProcessed/CSFn_WMn_Register/CSFn/case1')
# UI.mode = 0


if UI.mode == 0: register_cls(dir_in = UI.dir_in , dir_out = UI.dir_out , maskCrop='mask_t1').apply_register()
else:            register_cls(dir_in = UI.dir_in , dir_out = UI.dir_out , maskCrop='mask_t1').register_All()