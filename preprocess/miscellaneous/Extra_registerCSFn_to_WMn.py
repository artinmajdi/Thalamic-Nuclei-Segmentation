import os, sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from otherFuncs import smallFuncs
from preprocess import uncrop
from nilearn import image as niImage
import nibabel as nib
import numpy as np
from shutil import copyfile
# import ants


class UserEntry():
    def __init__(self):
        self.dir_in  = ''
        self.mode    = 0

        for en in range(len(sys.argv)):
            if sys.argv[en].lower() in ('-i','--input'):
                self.dir_in  = os.path.abspath(sys.argv[en+1])
            elif sys.argv[en].lower() in ('-m','--mode'):
                self.mode    = sys.argv[en+1]

class register_cls():
    def __init__(self, dir_in = '' , wmn_name='' , csfn_name=''):

        self.dir_in  = dir_in
        self.csfn_name = csfn_name
        self.wmn_name = wmn_name

    def apply_register(self):

        def warp_nuclei(self):
            smallFuncs.mkDir(self.dir_in + '/Label_registered')
            for nucleus in smallFuncs.Nuclei_Class().All_Nuclei().Names:
                IN  = self.dir_in  + '/Label/'            + nucleus + '.nii.gz'
                OUT = self.dir_in  + '/Label_registered/' + nucleus + '.nii.gz'
                csfn = self.dir_in + '/' + self.csfn_name + '.nii.gz'
                os.system('antsApplyTransforms -d 3 -i %s -r %s -o %s -t %s/aff0GenericAffine.mat -n NearestNeighbor' % (IN , csfn , OUT , self.dir_in+'/temp'))

        def warp_crop_mask(self):
            IN  = self.dir_in  + '/temp_wmn/CropMask.nii.gz'
            OUT = self.dir_in  + '/temp/CropMask.nii.gz'
            csfn = self.dir_in + '/' + self.csfn_name + '.nii.gz'
            os.system('antsApplyTransforms -d 3 -i %s -r %s -o %s -t %s/aff0GenericAffine.mat -n NearestNeighbor' % (IN , csfn , OUT , self.dir_in+'/temp'))

        IN  = self.dir_in +'/temp'
        csfn = self.dir_in + '/' + self.csfn_name + '.nii.gz'
        wmn = self.dir_in + '/' + self.wmn_name  + '.nii.gz'
        os.system('antsRegistration  -d 3 --float 0 --output \[ %s/aff,%s/affine.nii.gz\] -r \[ %s, %s,1\] -t Rigid\[0.1\] --metric MI\[ %s, %s,1,32,Regular,0.25\] --convergence \[1000x500x250x100,1e-7,10\] -v -f 8x4x2x1 -s 3x2x1x0vox -v -t Affine\[0.1\] --metric MI\[ %s, %s,1,32,Regular,0.25\] --convergence \[1000x500x250x100,5e-8,10\] -f 8x4x2x1 -s 3x2x1x0vox' % (IN,IN, csfn,wmn , csfn,wmn , csfn,wmn) )

        warp_nuclei(self)
        warp_crop_mask(self)

    def register_All(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'case' in s]:
            print(subj , '\n')
            dir_in  = self.dir_in + '/' + subj
            temp = register_cls(dir_in=dir_in, wmn_name=self.wmn_name , csfn_name=self.csfn_name )
            temp.apply_register()



UI = UserEntry()


if UI.mode == 'all': register_cls(dir_in = UI.dir_in , wmn_name='wmn' , csfn_name='csfn').register_All()
else:            register_cls(dir_in = UI.dir_in , wmn_name='wmn' , csfn_name='csfn').apply_register()