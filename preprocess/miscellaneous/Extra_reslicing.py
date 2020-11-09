import os, sys
from nilearn import image as niImage
import nibabel as nib
import json
import numpy as np
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from otherFuncs import smallFuncs


class UserEntry:
    def __init__(self):
        self.dir_in  = ''
        self.dir_out = ''
        self.mode    = 0

        for en in range(len(sys.argv)):
            if sys.argv[en].lower() in ('-i','--input'):    self.dir_in  = os.path.abspath(sys.argv[en+1])
            
            elif sys.argv[en].lower() in ('-o','--output'): self.dir_out = os.path.abspath(sys.argv[en+1])
            
            elif sys.argv[en].lower() in ('-m','--_mode'):   self.mode    = sys.argv[en+1]
            
        print(self.dir_in)
        print(self.dir_out)
        
class Reference:
    def __init__(self, nucleus='Image'): 

        self.dir_origRefImage = 'path-to-reference-case3_DS/'
        self.dir = '/general/Reslicing/'
        self.nucleus = nucleus if not ('.nii.gz' in nucleus) else nucleus.split('.nii.gz')[0]
    def write(self):
        
        if self.nucleus == 'Image': DirT = 'WMnMPRAGE_bias_corr.nii.gz'
        else: DirT = 'Label/' + self.nucleus + '.nii.gz'
            
        if os.path.exists(self.dir_origRefImage + DirT):
            ref = nib.load(self.dir_origRefImage + DirT)

            Info_Ref = {'affine':ref.affine.tolist() , 'shape':ref.shape}
            with open(self.dir + self.nucleus + '.json', "w") as j:
                j.write(json.dumps(Info_Ref))
        else:
            print('nucleus %s doesn not exist' % self.nucleus )

    def read(self):
        if os.path.exists(self.dir + self.nucleus + '.json'):
            
            with open(self.dir + self.nucleus + '.json', "r") as j: 
                info = json.load(j)

                info['affine'] = np.array(info['affine'])
                info['shape']  = tuple(info['shape']) 

                return info
        else:
            print('nucleus %s doesn not exist' % self.nucleus )            

    def write_all_nuclei(self):       
        for self.nucleus in np.append('Image' , smallFuncs.Nuclei_Class(method='Cascade').allNames): 
            Reference(self.nucleus).write()

class reslice_cls:
    def __init__(self, dir_in = '' , dir_out = ''):

        self.dir_in  = dir_in
        self.dir_out = dir_out

    def apply_reslice(self):
        
        def apply_to_Image(image , self):            
            ref = Reference(nucleus='Image').read()

            input_image  = self.dir_in  + '/' + image + '.nii.gz'
            output_image = self.dir_out + '/' + image + '.nii.gz'

            im = niImage.resample_img(img=nib.load(input_image), target_affine=ref['affine'][:3,:3] , interpolation='continuous') #  , target_shape=ref['shape'] 
            nib.save(im, output_image)        

        def apply_to_mask(nucleus , self):
            ref = Reference(nucleus=nucleus).read()
            
            input_image  = self.dir_in  + '/Label/' + nucleus + '.nii.gz'
            output_image = self.dir_out + '/Label/' + nucleus + '.nii.gz'

           
            # affine = np.zeros(3)
            # for f in range(3): affine[f,f] = ref['affine'][f,f]
            # msk = niImage.resample_img(img= nib.load(input_image), target_affine=affine[f,f]  , interpolation='nearest')  # , target_shape=ref['shape'] 
            msk = niImage.resample_img(img=nib.load(input_image) , target_affine=ref['affine'][:3,:3] , interpolation='nearest')  # , target_shape=ref['shape'] 
            nib.save(msk, output_image)
            
        smallFuncs.mkDir(self.dir_out)
        for image in [n.split('.nii.gz')[0] for n in os.listdir(self.dir_in) if '.nii.gz' in n]: 
            print('Applying to Image')
            apply_to_Image(image , self)
        
        smallFuncs.mkDir(self.dir_out + '/Label/')
        for nucleus in smallFuncs.Nuclei_Class(method='Cascade').allNames:  # [n for n in os.listdir(self.dir_in + '/Label/') if '.nii.gz' in n]: 
            print('Applying to',nucleus)
            try:
                apply_to_mask(nucleus , self)
            except Exception as e:
                print(e)

    def reslice_all(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'case' in s]:
            print(subj , '\n')
            dir_in  = self.dir_in + '/' + subj
            dir_out = self.dir_out + '/' + subj
            temp = reslice_cls(dir_in=dir_in , dir_out=dir_out)
            temp.apply_reslice()



UI = UserEntry()

if UI.mode == 'all': reslice_cls(dir_in = UI.dir_in , dir_out = UI.dir_out).reslice_all()
else: reslice_cls(dir_in = UI.dir_in , dir_out = UI.dir_out).apply_reslice()
