import nibabel

dir = '/array/ssd/msmajdi/data/preProcessed/MS_WMn_15T/step5_flipped_back_to_original/vimp2_pre/'


msk1 = nibabel.load(dir + 'Predictions_Right/AllLabels.nii.gz')
msk2 = nibabel.load(dir + 'Predictions_Left/AllLabels.nii.gz')

msk3 = nibabel.Nifti1Image( msk1.get_data() + msk2.get_data() ,header=msk1.header , affine=msk1.affine)
nibabel.save(msk3,dir + 'AllLabels_Left_and_Right.nii.gz')