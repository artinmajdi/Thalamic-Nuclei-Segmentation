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

params = paramFunc.Run(UserInfo.__dict__, terminal=True)



dirr = '/array/ssd/msmajdi/experiments/keras/exp1/results/sE11_HCascade_FM20_7T_Main/sd2/vimp2_915_07112013_LC_MS/'
# def Save_AllNuclei_inOne():

A = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()

im = nib.load( dirr + '1-THALAMUS.nii.gz')   
Mask = []
for cnt , name in zip(A.Indexes , A.Names):                                
    if cnt != 1:
        msk = nib.load( dirr + name  + '.nii.gz' ).get_data()  
        Mask = cnt*msk if Mask == [] else Mask + cnt*msk   
    

smallFuncs.saveImage( Mask , im.affine , im.header, '/array/ssd/msmajdi/AllLabels.nii.gz')