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
            if sys.argv[en].lower() in ('-i','--input'):    self.dir_in  = os.getcwd() + '/' + sys.argv[en+1] if '/array/ssd' not in sys.argv[en+1] else sys.argv[en+1]
            elif sys.argv[en].lower() in ('-o','--output'): self.dir_out = os.getcwd() + '/' + sys.argv[en+1] if '/array/ssd' not in sys.argv[en+1] else sys.argv[en+1]
            elif sys.argv[en].lower() in ('-m','--mode'):   self.mode    = sys.argv[en+1]
            
class register_cls():
    def __init__(self, dir_in = '' , dir_out = '' , wmn_name='' , csfn_name=''):

        self.dir_in  = dir_in
        self.dir_out = dir_out
        self.csfn_name = csfn_name
        self.wmn_name = wmn_name

    def apply_register(self):

        def warp_nuclei(self):
            smallFuncs.mkDir(self.dir_out + '/Label') 
            for nucleus in smallFuncs.Nuclei_Class(method='Cascade').allNames:
                IN  = self.dir_in  + '/Label/' + nucleus    + '_PProcessed.nii.gz'
                OUT = self.dir_out + '/Label/' + nucleus    + '_PProcessed.nii.gz'        
                csfn = self.dir_in + '/' + self.csfn_name + '.nii.gz'    
                os.system('antsApplyTransforms -d 3 -i %s -r %s -o %s -t %s/aff0GenericAffine.mat -n NearestNeighbor' % (IN , csfn , OUT , self.dir_in)) 
                os.system('cp -r %s %s/temp %s/ ' %(csfn , self.dir_in , self.dir_out))

        def warp_AV_Mask(self):
            smallFuncs.mkDir(self.dir_out + '/temp') 
            IN  = self.dir_in  + '/temp/CropMask_AV.nii.gz'
            OUT = self.dir_out + '/temp/CropMask_AV.nii.gz'            
            os.system('antsApplyTransforms -d 3 -i %s -r %s/crop_t1.nii.gz -o %s -t %s/aff0GenericAffine.mat -n NearestNeighbor' % (IN , self.dir_in , OUT , self.dir_in))  


        IN  = self.dir_in  
        csfn = self.dir_in + '/' + self.csfn_name + '.nii.gz'
        wmn = self.dir_in + '/' + self.wmn_name  + '.nii.gz'
        os.system('antsRegistration  -d 3 --float 0 --output \[ %s/aff,%s/affine.nii.gz\] -r \[ %s, %s,1\] -t Rigid\[0.1\] --metric MI\[ %s, %s,1,32,Regular,0.25\] --convergence \[1000x500x250x100,1e-7,10\] -v -f 8x4x2x1 -s 3x2x1x0vox -v -t Affine\[0.1\] --metric MI\[ %s, %s,1,32,Regular,0.25\] --convergence \[1000x500x250x100,5e-8,10\] -f 8x4x2x1 -s 3x2x1x0vox' % (IN,IN, csfn,wmn , csfn,wmn , csfn,wmn) )  
        # os.system('antsRegistration  -d 3 --float 0 --output \[ %s/aff,%s/affine.nii.gz\] -r \[ %s/crop_t1.nii.gz, %s/crop_wmn.nii.gz,1\] -t Rigid\[0.1\] --metric MI\[ %s/crop_t1.nii.gz, %s/crop_wmn.nii.gz,1,32,Regular,0.25\] --convergence \[1000x500x250x100,1e-7,10\] -v -f 8x4x2x1 -s 3x2x1x0vox -v -t Affine\[0.1\] --metric MI\[ %s/crop_t1.nii.gz, %s/crop_wmn.nii.gz,1,32,Regular,0.25\] --convergence \[1000x500x250x100,5e-8,10\] -f 8x4x2x1 -s 3x2x1x0vox' % (IN,IN, csfn,wmn , csfn,wmn , csfn,wmn) )  
        
        warp_nuclei(self)
        # warp_AV_Mask(self)

    def register_All(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'case' in s]:
            print(subj , '\n')
            dir_in  = self.dir_in + '/' + subj
            dir_out = self.dir_out + '/' + subj
            temp = register_cls(dir_in=dir_in , dir_out=dir_out, wmn_name=self.wmn_name , csfn_name=self.csfn_name )
            temp.apply_register()




# dir_in  = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn_Orig/WMn/case1'
# dir_out = smallFuncs.mkDir('/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/WMn/case1')

UI = UserEntry()
# UI.dir_in  = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset3_new_ctrl_ms_csfn/CNN/CNN_wo_N4/manual_Labels/csfn_step0/csfn_with_wmn'
# UI.dir_out = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset3_new_ctrl_ms_csfn/CNN/CNN_wo_N4/manual_Labels/csfn_step1_registered'
# UI.mode = 'all'


if UI.mode == 'all': register_cls(dir_in = UI.dir_in , dir_out = UI.dir_out , wmn_name='crop_wmn' , csfn_name='PProcessed').register_All()
else:            register_cls(dir_in = UI.dir_in , dir_out = UI.dir_out , wmn_name='crop_wmn' , csfn_name='PProcessed').apply_register()