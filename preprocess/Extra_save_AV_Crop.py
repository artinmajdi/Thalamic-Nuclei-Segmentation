import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import otherFuncs.smallFuncs as smallFuncs
from shutil import copyfile


class UserEntry():
    def __init__(self):
        self.dir_in  = ''
        self.dir_out = ''
        self.mode    = 0

        class Template:
            def __init__(self, dirT = '/array/ssd/msmajdi/code/general/RigidRegistration'):
                self.Image = dirT + '/origtemplate.nii.gz' 
                self.Mask  = dirT + '/CropMask_AV.nii.gz' 

        for en in range(len(sys.argv)):
            if sys.argv[en].lower() in ('-i','--input'):      self.dir_in   = os.getcwd() + '/' + sys.argv[en+1]
            elif sys.argv[en].lower() in ('-t','--template'): self.Template = Template(os.getcwd() + '/' + sys.argv[en+1])
            elif sys.argv[en].lower() in ('-m','--mode'):     self.mode     = int(sys.argv[en+1])                     
                    

    

class AV_crop():
    def __init__(self, UI = ''):

        self.dir_in   = UI.dir_in
        self.Template = UI.Template

    def apply_Register(UI):
        
        a = next(os.walk(UI.dir_in))

        Image = UI.dir_in + '/' + [s for s in a[2] if not('PProcessed.nii.gz' in s)][0]
        outMask = UI.dir_in + 'temp/CropMask_AV.nii.gz'
        LinearAffine = UI.dir_in + 'temp/deformation'

        if not os.path.exists(outMask):
            if not os.path.isfile(LinearAffine + '/linearAffine.txt'): 
                os.system("ANTS 3 -m CC[%s, %s ,1,5] -o %s -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(Image , UI.Template.Image , LinearAffine + '/linear') )
            
            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(UI.Template.Mask , outMask , Image , LinearAffine) )

    def loop_all_subjects(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'vimp' in s]:
            print(subj , '\n')
            UI.dir_in  = self.dir_in + '/' + subj

            temp = AV_crop(UI)
            temp.apply_Register()




UI = UserEntry()
UI.dir_in  = '/array/ssd/msmajdi/experiments/keras/exp5_CSFn/train/Main/vimp2_668_02282013_CD'
UI.mode = 0

AV_crop(UI).apply_Register()
# if UI.mode == 0: reslice_cls(dir_in = UI.dir_in , dir_out = UI.dir_out).apply_reslice()
# else:            reslice_cls(dir_in = UI.dir_in , dir_out = UI.dir_out).reslice_all()