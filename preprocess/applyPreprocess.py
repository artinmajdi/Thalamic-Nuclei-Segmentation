import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
from preprocess import Extra_AV_Crop, augmentA, BashCallingFunctionsA, normalizeA, croppingA
# from preprocess import  # create_AV_Mask
from nilearn import image as niImage
import nibabel as nib
import json
from shutil import copyfile
import numpy as np

# TODO  check 3T 7T dimension and interpolation
# TODO check image format and convert to nifti

#! mode: 1: on train & test folders in the experiment
#! mode: 2: on individual image
def main(params, mode):

    def loopOverSubjects_PreProcessing(params, Mode):
        
        class Info:
            mode = Mode
            dirr = params.directories.Train if Mode == 'train' else params.directories.Test
            Length = len(dirr.Input.Subjects)
            subjectName = ''
            ind = ''
            Subjects = dirr.Input.Subjects

        for Info.ind, Info.subjectName in enumerate(Info.Subjects): apply_On_Individual(params, Info)

    if params.preprocess.Mode:
        params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
        if not params.preprocess.TestOnly: loopOverSubjects_PreProcessing(params, 'train')

    loopOverSubjects_PreProcessing(params, 'test')

def apply_On_Individual(params,Info):

    subject = Info.Subjects[Info.subjectName]

    if 1: print( '(' + str(Info.ind) + '/'+str(Info.Length) + ')' , Info.mode, Info.subjectName)   

    BashCallingFunctionsA.BiasCorrection( subject , params)

    apply_reslice(subject , params)
    
    if ('Aug' not in Info.subjectName) and ('CSFn' not in Info.subjectName): 
        BashCallingFunctionsA.RigidRegistration( subject , params.WhichExperiment.HardParams.Template , params.preprocess)        
        croppingA.main(subject , params)

    # Extra_AV_Crop.main(dir_in=subject.address, dir_template=params.WhichExperiment.HardParams.Template.Address)
    # BashCallingFunctionsA.RigidRegistration_2AV( subject , params.WhichExperiment.HardParams.Template , params.preprocess)
    # croppingA.crop_AV(subject , params)

    return params

def apply_Augmentation(params):
    augmentA.main_augment( params , 'Linear' , 'experiment')
    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    augmentA.main_augment( params , 'NonLinear' , 'experiment')


def apply_reslice(subject, params):

    
    class Reference():
        def __init__(self, nucleus='Image'): 

            self.dir_origRefImage = '/array/ssd/msmajdi/experiments/keras/exp3/train/Main/vimp2_819_05172013_DS/'
            self.dir = '/array/ssd/msmajdi/code/thalamus/keras/general/Reslicing/'
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
            for self.nucleus in np.append('Image' , smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Names): 
                Reference(self.nucleus).write()

    def apply_reslicing_main(input_image, output_image, outDebug, interpolation , ref):

        if os.path.isfile(outDebug):
            copyfile(outDebug , output_image)
        else:
            im = niImage.resample_img(img=nib.load(input_image), target_affine=ref['affine'][:3,:3] , interpolation='continuous') #  , target_shape=ref['shape'] 
            # im = niImage.resample_img(img=nib.load(input_image), target_affine=ref['affine'] , interpolation='continuous', target_shape=nib.load(input_image).shape) #  
            nib.save(im, output_image) 
            copyfile(output_image , outDebug)


    def apply_to_Image(subject):            
        ref = Reference(nucleus='Image').read()

        input_image  = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
        output_image = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
        outDebug = smallFuncs.mkDir(subject.Temp.address + '/')  + subject.ImageOriginal + '_resliced.nii.gz'

        apply_reslicing_main(input_image, output_image, outDebug, 'continuous' , ref)       

    def apply_to_mask(subject):
        ref = Reference(nucleus=nucleus).read()
        
        if subject.Label.address:
            input_nucleus  = subject.Label.address + '/' + nucleus + '_PProcessed.nii.gz'
            output_nucleus = subject.Label.address + '/' + nucleus + '_PProcessed.nii.gz'
            outDebug  = smallFuncs.mkDir(subject.Label.address + '/temp/') + nucleus + '_resliced.nii.gz'

            apply_reslicing_main(input_nucleus, output_nucleus, outDebug, 'nearest' , ref)    

    if params.preprocess.Reslicing.Mode and params.preprocess.Mode:
        print('     ReSlicing') 
        apply_to_Image(subject)
        for nucleus in smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Names:  apply_to_mask(subject)

