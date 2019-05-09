import numpy as np
# import os, sys
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# # sys.path.append( os.path.dirname(os.path.dirname(__file__)) )
# import otherFuncs.smallFuncs as smallFuncs
# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# import keras
# import h5py
# import nibabel as nib
# import nilearn 
# from nilearn import image as niImage

# params = paramFunc.Run(UserInfo.__dict__, terminal=True)

a = [['a',1],['b',2]
b = [['e',3] , ['b',2]]
c = np.concatenate((a,b), axis=0)
print(np.unique(c))
print(list(set(a+b)))
# def func_reslice(dir1, dir_ref, interpolation):
#     im = niImage.load_img(dir1)
#     min , max = im.get_data().min() , im.get_data().max()
#     print(min , max)
#     if max >1:
#         print('---')
#     ref = niImage.load_img(dir_ref)
#     imRL = niImage.resample_img(img=im , target_affine=ref.affine  , target_shape=im.shape,interpolation=interpolation)    
#     return imRL

# dir_ref = '/array/ssd/msmajdi/experiments/keras/exp3/train/Main/vimp2_819_05172013_DS/'
# dir_in  = '/array/ssd/msmajdi/data/preProcessed/7T/vimp2_A/'
# dir_out = '/array/ssd/msmajdi/data/preProcessed/7T/vimp2_A_b/'
# smallFuncs.mkDir(dir_out + 'Label/')

# imRL = func_reslice(dir_in + 'WMnMPRAGE_bias_corr.nii.gz', dir_ref + 'WMnMPRAGE_bias_corr.nii.gz' , 'continuous')
# nib.save(imRL , dir_out + 'WMnMPRAGE_bias_corr.nii.gz')

# for label in smallFuncs.Nuclei_Class(method = 'Cascade').All_Nuclei().Names:
#     print(label)
#     imRL = func_reslice(dir_in + 'Label/' + label + '.nii.gz', dir_ref + 'Label/' + label + '.nii.gz' , 'nearest')
#     nib.save(imRL , dir_out + 'Label/' + label + '.nii.gz')

# print('---')