import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import otherFuncs.smallFuncs as smallFuncs
from preprocess import Extra_AV_Crop, augmentA, BashCallingFunctionsA, normalizeA, croppingA
from nilearn import image as niImage
import nibabel as nib
import json
from shutil import copyfile
import numpy as np


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
    return params

def apply_Augmentation(params):
    augmentA.main_augment( params , 'Linear' , 'experiment')
    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    augmentA.main_augment( params , 'NonLinear' , 'experiment')

def apply_reslice(subject, params):

    class Reference():
        def __init__(self, nucleus='Image'): 

            self.dir_origRefImage = '/array/ssd/msmajdi/experiments/keras/exp3/train/Main/vimp2_819_05172013_DS/'
            self.dir = params.WhichExperiment.Experiment.code_address + '/general/Reslicing/'
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

def RigidRegistration(subject , Template , preprocess):

    processed = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.Temp.address + '/CropMask.nii.gz'
    LinearAffine = subject.Temp.Deformation.address + '/linearAffine.txt'
    if preprocess.Mode and preprocess.Cropping.Mode: # and not os.path.isfile(outP):
        print('     Rigid Registration')
        if not os.path.isfile(LinearAffine): 
            os.system("ANTS 3 -m CC[%s, %s ,1,5] -o %s -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(processed , Template.Image , subject.Temp.Deformation.address + '/linear') )

        if not os.path.isfile(outP): 
            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Mask , outP , processed , LinearAffine) )

def BiasCorrection(subject , params):
    
    inP  = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_bias_corr.nii.gz'
    if params.preprocess.Mode and params.preprocess.BiasCorrection.Mode:
        if os.path.isfile(outDebug) and params.preprocess.Debug.justForNow:
            copyfile(outDebug , outP)
        else:
            print('     Bias Correction')            
            os.system( "N4BiasFieldCorrection -d 3 -i %s -o %s -b [200] -s 3 -c [50x50x30x20,1e-6]"%( inP, outP )  )
            if params.preprocess.Debug.doDebug:
                copyfile(outP , outDebug)

"""
def RigidRegistration_2AV(subject , Template , preprocess):
    
    
    processed = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    FullImage = subject.address + '/' + subject.ImageOriginal + '.nii.gz'
    
    
    LinearAffine_FullImage = subject.Temp.Deformation.address 
    LinearAffine_CropImage = subject.Temp.Deformation.address  + '_Crop'
    
    
    Template_CropImage = Template.Address + 'cropped_origtemplate.nii.gz'
    # Template_FullImage = Template.Address + 'origtemplate.nii.gz'
    
    outP_crop = subject.Temp.address + '/CropMask_AV.nii.gz'
    if preprocess.Cropping.Mode and not os.path.isfile(outP_crop):  
        
        if not os.path.isfile(LinearAffine_FullImage + '/linearAffine.txt' ): 
            print('     Rigid Registration of cropped Image')
            smallFuncs.mkDir(LinearAffine_CropImage)
            os.system("ANTS 3 -m CC[%s, %s ,1,5] -o %s -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(processed , Template_CropImage , LinearAffine_CropImage + '/linear') )
        
            print('     Warping of cropped Mask')
            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Address + 'CropMask_AV.nii.gz' , outP_crop , processed , LinearAffine_CropImage) )

        else:
            outP_full = subject.Temp.address + '/Mask_AV.nii.gz'
            mainCrop = subject.Temp.address + '/CropMask.nii.gz.nii.gz'

            print('     Warping of full Mask')
            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Address + 'Mask_AV.nii.gz' , outP_full , FullImage , LinearAffine_FullImage) )
            os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Address + 'CropMaskV3.nii.gz' , outP_full , FullImage , LinearAffine_FullImage) )

            print('    Cropping the Full AV Mask')
            cropping_AV_Mask(outP_full, outP_crop, mainCrop)
"""

def Bash_AugmentNonLinear(subject , subjectRef , outputAddress): # Image , Mask , Reference , output):

    print('     Augment NonLinear')

    ImageOrig = subject.address       + '/' + subject.ImageProcessed + '.nii.gz'
    # MaskOrig  = subject.Label.address + '/' + subject.Label.LabelProcessed + '.nii.gz'
    ImageRef  = subjectRef.address    + '/' + subjectRef.ImageProcessed + '.nii.gz'

    OutputImage = outputAddress  + '/' + subject.ImageProcessed + '.nii.gz'
    labelAdd    = smallFuncs.mkDir(outputAddress + '/Label')
    deformationAddr = smallFuncs.mkDir(outputAddress + '/Temp/deformation')


    if not os.path.isfile(OutputImage):
        os.system("ANTS 3 -m CC[%s, %s,1,5] -t SyN[0.25] -r Gauss[3,0] -o %s -i 30x90x20 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000"%(ImageOrig , ImageRef , deformationAddr + '/test') )
        os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(ImageOrig , OutputImage , ImageOrig , deformationAddr + '/testWarp.nii.gz') )

    _, _, names = smallFuncs.NucleiSelection(ind = 1)
    for name in names:
        MaskOrig  = subject.Label.address + '/' + name + '_PProcessed.nii.gz'
        OutputMask  = labelAdd + '/' + name + '_PProcessed.nii.gz'
        if not os.path.isfile(OutputMask):
            os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(MaskOrig , OutputMask , MaskOrig , deformationAddr + '/testWarp.nii.gz' ) )

"""
def cropping_AV_Mask(inP, outP, crop):
    
    def cropImage_FromCoordinates(CropMask , Gap): 
        BBCord = smallFuncs.findBoundingBox(CropMask>0.5)

        d = np.zeros((3,2),dtype=np.int)
        for ix in range(len(BBCord)):
            d[ix,:] = [  BBCord[ix][0]-Gap[ix] , BBCord[ix][-1]+Gap[ix]  ]
            d[ix,:] = [  max(d[ix,0],0)    , min(d[ix,1],CropMask.shape[ix])  ]

        return d
            
    # crop = subject.Temp.address + '/CropMask.nii.gz' 
    
    d = cropImage_FromCoordinates(crop , [0,0,0])
    mskC = nib.load(inP).slicer[ d[0,0]:d[0,1], d[1,0]:d[1,1], d[2,0]:d[2,1] ]                    
    nib.save(mskC , outP)
"""