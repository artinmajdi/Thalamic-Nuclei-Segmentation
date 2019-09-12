# import nilearn
import nibabel

dir = '/array/ssd/msmajdi/experiments/keras/exp6/results/for_paper/'


msk1 = nibabel.load(dir + 'AllLabels.nii.gz')
msk2 = nibabel.load(dir + 'AllLabels_flippedBack.nii.gz')

msk3 = nibabel.Nifti1Image( msk1.get_data() + msk2.get_data() ,header=msk1.header , affine=msk1.affine)
nibabel.save(msk3,dir + 'merged.nii.gz')

