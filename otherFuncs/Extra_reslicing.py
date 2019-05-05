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

    
def func_reslice(dir1, dir_ref, interpolation):

    im  = niImage.load_img(dir1)
    ref = niImage.load_img(dir_ref)

    if (interpolation == 'nearest'): im = func_FixMinMax(im)
    
    return niImage.resample_img(img=im , target_affine=ref.affine  , target_shape=im.shape,interpolation=interpolation)  

dir_ref = '/array/ssd/msmajdi/experiments/keras/exp3/train/Main/vimp2_819_05172013_DS/'

dir_ET_in  = '/array/ssd/msmajdi/data/preProcessed/ET_orig/'
dir_ET_out = smallFuncs.mkDir('/array/ssd/msmajdi/data/preProcessed/ET_Resliced/')


for sT in ['7T/' , '3T/']:
    for subj in [s for s in os.listdir(dir_ET_in + sT) if 'vimp' in s]:
        
        dir_in  = dir_ET_in + sT + subj + '/'
        dir_out = smallFuncs.mkDir(dir_ET_out + sT + subj + '/')


        smallFuncs.mkDir(dir_out + 'Label/')

        imRL = func_reslice(dir_in + 'WMnMPRAGE_bias_corr.nii.gz', dir_ref + 'WMnMPRAGE_bias_corr.nii.gz' , 'continuous')
        nib.save(imRL , dir_out + 'WMnMPRAGE_bias_corr.nii.gz')

        for label in smallFuncs.Nuclei_Class(method = 'Cascade').All_Nuclei().Names:
            print(subj , label)
                            
            imRL = func_reslice(dir_in + 'Label/' + label + '.nii.gz', dir_ref + 'Label/' + label + '.nii.gz' , 'nearest')
            nib.save(imRL , dir_out + 'Label/' + label + '.nii.gz')

        print('---')