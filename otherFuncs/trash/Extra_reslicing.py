import numpy as np
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append( os.path.dirname(os.path.dirname(__file__)) )
import otherFuncs.smallFuncs as smallFuncs
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
# import keras
# import h5py
import nibabel as nib
import nilearn 
from nilearn import image as niImage

# params = paramFunc.Run(UserInfo.__dict__, terminal=True)

def func_FixMinMax(im):

    imD = im.get_data()
    Min , Max = imD.min() , imD.max()
        
    if Max > 1:
        affine, header = im.affine , im.header
        imD = (imD - Min) / (Max-Min)        
        # im = nib.Nifti1Image(imD.astype(im.get_data_dtype().name),affine)
        im = nib.Nifti1Image(imD.astype('float32'), affine)
        im.get_header = header        
    return im
    
def findingNot01Labels(dir_ET_in , sT):    
    for subj in [s for s in os.listdir(dir_ET_in + sT) if 'vimp' in s]:
        
        dir_in  = dir_ET_in + sT + subj + '/'

        for label in smallFuncs.Nuclei_Class(method = 'Cascade').All_Nuclei().Names:
            
            msk  = nib.load(dir_in + 'Label/' + label + '.nii.gz')
            
            print(sT, subj , label, msk.get_data().max()) if msk.get_data().max() > 1 else print(sT, subj , label)

def loopOverAllSubjects(dir_ET_in , dir_ET_out , dir_ref , sT , MnMx_Flag , RL_Flag , targetShape_Mode):
    def func_reslice(dir1, dir_ref, interpolation):


        im  = niImage.load_img(dir1)
        ref = niImage.load_img(dir_ref)

        if MnMx_Flag and (interpolation == 'nearest'): im = func_FixMinMax(im)

        if targetShape_Mode == 'ref':     target_shape = ref.shape  
        elif targetShape_Mode == 'input': target_shape = im.shape
        if RL_Flag: im = niImage.resample_img(img=im , target_affine=ref.affine  , target_shape=target_shape , interpolation=interpolation) 
                    
        return im
   
    def func_apply_reslice_perSubj(dir_in ,  dir_out , subj):

        imRL = func_reslice(dir_in + 'WMnMPRAGE_bias_corr.nii.gz', dir_ref + 'WMnMPRAGE_bias_corr.nii.gz' , 'continuous')
        nib.save(imRL , dir_out + 'WMnMPRAGE_bias_corr.nii.gz')

        smallFuncs.mkDir(dir_out + 'Label/')
        for label in smallFuncs.Nuclei_Class(method = 'Cascade').All_Nuclei().Names:
            print(sT , subj , label)
                            
            imRL = func_reslice(dir_in + 'Label/' + label + '.nii.gz', dir_ref + 'Label/' + label + '.nii.gz' , 'nearest')
            nib.save(imRL , dir_out + 'Label/' + label + '.nii.gz')
                            
    for subj in [s for s in os.listdir(dir_ET_in + sT) if 'vimp' in s]:
        
        dir_in  = dir_ET_in + sT + subj + '/'
        dir_out = smallFuncs.mkDir(dir_ET_out + sT + subj + '/')
        
        func_apply_reslice_perSubj(dir_in , dir_out , subj)

dir_ref    = '/array/ssd/msmajdi/experiments/keras/exp3/train/Main/vimp2_819_05172013_DS/'
dir_ET_in  = '/array/ssd/msmajdi/data/preProcessed/ET_orig/'
dir_ET_out = smallFuncs.mkDir('/array/ssd/msmajdi/data/preProcessed/ET_Resliced/')


# loopOverAllSubjects(dir_ET_in , dir_ET_out , dir_ref , '7T/' , 1 , 1 , 'input')
loopOverAllSubjects(dir_ET_in , dir_ET_out , dir_ref , '3T/' , 1 , 1 , 'input')


# for sT in ['7T/'  , '3T/']:
#     findingNot01Labels(dir_ET_in , sT)
#     # findingNot01Labels(dir_ET_out , sT)