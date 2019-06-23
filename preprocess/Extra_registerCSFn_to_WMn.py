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
    def __init__(self, dir_in = '' , dir_out = ''):

        self.dir_in  = dir_in
        self.dir_out = dir_out

    def apply_register(self):

        def warp_nuclei(self):
            smallFuncs.mkDir(self.dir_out + '/Label') 
            for nucleus in smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Names:
                IN  = self.dir_in  + '/Label/' + nucleus    + '.nii.gz'
                OUT = self.dir_out + '/Label/' + nucleus    + '.nii.gz'            
                os.system('antsApplyTransforms -d 3 -i %s -r %s/crop_t1.nii.gz -o %s -t %s/aff0GenericAffine.mat -n NearestNeighbor' % (IN , self.dir_in , OUT , self.dir_in))  
                os.system('cp -r %s/crop_t1.nii.gz %s/temp %s/ ' %(self.dir_in , self.dir_in , self.dir_out))

        def warp_AV_Mask(self):
            smallFuncs.mkDir(self.dir_out + '/temp') 
            IN  = self.dir_in  + '/temp/CropMask_AV.nii.gz'
            OUT = self.dir_out + '/temp/CropMask_AV.nii.gz'            
            os.system('antsApplyTransforms -d 3 -i %s -r %s/crop_t1.nii.gz -o %s -t %s/aff0GenericAffine.mat -n NearestNeighbor' % (IN , self.dir_in , OUT , self.dir_in))  


        IN  = self.dir_in  
        os.system('antsRegistration  -d 3 --float 0 --output \[ %s/aff,%s/affine.nii.gz\] -r \[ %s/crop_t1.nii.gz, %s/crop_wmn.nii.gz,1\] -t Rigid\[0.1\] --metric MI\[ %s/crop_t1.nii.gz, %s/crop_wmn.nii.gz,1,32,Regular,0.25\] --convergence \[1000x500x250x100,1e-7,10\] -v -f 8x4x2x1 -s 3x2x1x0vox -v -t Affine\[0.1\] --metric MI\[ %s/crop_t1.nii.gz, %s/crop_wmn.nii.gz,1,32,Regular,0.25\] --convergence \[1000x500x250x100,5e-8,10\] -f 8x4x2x1 -s 3x2x1x0vox' % (IN,IN,IN,IN,IN,IN,IN,IN) )
        warp_nuclei(self)
        warp_AV_Mask(self)

    def register_All(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'vimp' in s]:
            print(subj , '\n')
            dir_in  = self.dir_in + '/' + subj
            dir_out = self.dir_out + '/' + subj
            temp = register_cls(dir_in=dir_in , dir_out=dir_out)
            temp.apply_register()




# dir_in  = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn_Orig/WMn/case1'
# dir_out = smallFuncs.mkDir('/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/WMn/case1')

UI = UserEntry()
# dir = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset2_with_Manual_Labels/full_Image/pre-steps/CSFn1/'
# UI.dir_in  = dir + 'step3_Cropped'
# UI.dir_out = smallFuncs.mkDir(dir + 'step4_registered')
# UI.mode = 'all'


if UI.mode == 'all': register_cls(dir_in = UI.dir_in , dir_out = UI.dir_out).register_All()
else:            register_cls(dir_in = UI.dir_in , dir_out = UI.dir_out).apply_register()