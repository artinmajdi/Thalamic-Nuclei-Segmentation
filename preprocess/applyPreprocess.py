import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import otherFuncs.smallFuncs as smallFuncs
from nilearn import image as niImage
import nibabel as nib
import json
from shutil import copyfile
import numpy as np


def main(params):

    def loop_subjects(params, Mode):
        
        class Info:
            mode = Mode
            dirr = params.directories.Train if Mode == 'train' else params.directories.Test
            Length = len(dirr.Input.Subjects)
            subjectName = ''
            ind = ''
            Subjects = dirr.Input.Subjects

        for Info.ind, Info.subjectName in enumerate(Info.Subjects): 
            apply_On_Individual(params, Info)

    if params.preprocess.Mode:
        # params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
        if not params.WhichExperiment.TestOnly:
            loop_subjects(params, 'train')

        loop_subjects(params, 'test')

def apply_On_Individual(params,Info):

    print( '(' + str(Info.ind) + '/'+str(Info.Length) + ')' , Info.mode, Info.subjectName)   

    if ('Aug' not in Info.subjectName): 
        subject = Info.Subjects[Info.subjectName]

        if params.preprocess.BiasCorrection: 
            print('     Bias Correction')
            BiasCorrection( subject , params)
    
        if params.preprocess.Cropping:
            print('     Rigid Registration')
            RigidRegistration( subject , params.WhichExperiment.HardParams.Template)        
        
            print('     Cropping')
            func_cropImage(params, subject)

        if params.preprocess.Reslicing:
            print('     ReSlicing') 
            apply_reslice(subject , params)

    return params

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
            for self.nucleus in np.append('Image' , smallFuncs.Nuclei_Class(method='Cascade').allNames): 
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
            outDebug  = subject.Label.Temp.address + '/' + nucleus + '_resliced.nii.gz'

            apply_reslicing_main(input_nucleus, output_nucleus, outDebug, 'nearest' , ref)    

    apply_to_Image(subject)
    for nucleus in smallFuncs.Nuclei_Class().Names:  
        apply_to_mask(subject)

def RigidRegistration(subject , Template):

    processed = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.Temp.address + '/CropMask.nii.gz'
    LinearAffine = subject.Temp.Deformation.address + '/linearAffine.txt'

    if not os.path.isfile(LinearAffine): 
        os.system("ANTS 3 -m CC[%s, %s ,1,5] -o %s -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(processed , Template.Image , subject.Temp.Deformation.address + '/linear') )

    if not os.path.isfile(outP): 
        os.system("WarpImageMultiTransform 3 %s %s -R %s %s"%(Template.Mask , outP , processed , LinearAffine) )

def BiasCorrection(subject , params):
    
    inP  = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_bias_corr.nii.gz'

    if os.path.isfile(outDebug):
        copyfile(outDebug , outP)
    else:
        os.system( "N4BiasFieldCorrection -d 3 -i %s -o %s -b [200] -s 3 -c [50x50x30x20,1e-6]"%( inP, outP )  )
        if params.preprocess.save_debug_files:
            copyfile(outP , outDebug)

def func_cropImage(params, subject):
    
    def cropImage_FromCoordinates(CropMask , Gap): 
        BBCord = smallFuncs.findBoundingBox(CropMask>0.5)

        d = np.zeros((3,2),dtype=np.int)
        for ix in range(len(BBCord)):
            d[ix,:] = [  BBCord[ix][0]-Gap[ix] , BBCord[ix][-1]+Gap[ix]  ]
            d[ix,:] = [  max(d[ix,0],0)    , min(d[ix,1],CropMask.shape[ix])  ]

        return d
            
    crop = subject.Temp.address + '/CropMask.nii.gz' 
    
    def check_crop(inP, outP, outDebug, CropCoordinates):

        def applyCropping(image):
            d = CropCoordinates
            return image.slicer[ d[0,0]:d[0,1], d[1,0]:d[1,1], d[2,0]:d[2,1] ]
                            
        if os.path.isfile(outDebug): 
            copyfile(outDebug , outP)
        else:
            if 'ANTs' in params.preprocess.Cropping.Method:
                os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

            elif 'python' in params.preprocess.Cropping.Method:
                if os.path.isfile(inP):
                    mskC = applyCropping( nib.load(inP))
                    nib.save(mskC , outP)
                
            if params.preprocess.save_debug_files: copyfile(outP , outDebug)
                        
    def directoriesImage(subject):
        inP  = outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'                  
        outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_Cropped.nii.gz'
        return inP, outP, outDebug 

    def directoriesNuclei(subject, ind):
        NucleusName, _, _ = smallFuncs.NucleiSelection(ind )
        inP = outP = subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz'
        outDebug = subject.Label.Temp.address + '/' + NucleusName + '_Cropped.nii.gz'
        return inP, outP, outDebug

    inP, outP, outDebug = directoriesImage(subject)          
    CropCoordinates = '' if os.path.isfile(outDebug) else cropImage_FromCoordinates(nib.load(crop).get_data() , [0,0,0])  
    check_crop(inP, outP, outDebug, CropCoordinates)

    for ind in params.WhichExperiment.Nucleus.FullIndexes:
        inP, outP, outDebug = directoriesNuclei(subject, ind)
        if not os.path.isfile(outDebug):
            if not CropCoordinates:
                CropCoordinates = cropImage_FromCoordinates(nib.load(crop).get_data() , [0,0,0])  

        check_crop(inP, outP, outDebug, CropCoordinates)


"""
def RigidRegistration_2AV(subject , Template , preprocess):
    
    
    processed = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    FullImage = subject.address + '/' + subject.ImageOriginal + '.nii.gz'
    
    
    LinearAffine_FullImage = subject.Temp.Deformation.address 
    LinearAffine_CropImage = subject.Temp.Deformation.address  + '_Crop'
    
    
    Template_CropImage = Template.Address + 'cropped_origtemplate.nii.gz'
    # Template_FullImage = Template.Address + 'origtemplate.nii.gz'
    
    outP_crop = subject.Temp.address + '/CropMask_AV.nii.gz'
    if preprocess.Cropping and not os.path.isfile(outP_crop):  
        
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