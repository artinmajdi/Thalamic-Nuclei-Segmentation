import nibabel
import os


dir_f = '/array/ssd/msmajdi/experiments/keras/exp6/results/sE12_Cascade_FM00_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_Best_w20priors_flippedThalmaus/2.5D_MV/'
dir = '/array/ssd/msmajdi/experiments/keras/exp6/results/sE12_Cascade_FM00_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_Best_w20priors/2.5D_MV/temp/'


for v in [s for s in os.listdir(dir_f) if 'case' in s]:

    print('subject:',v)
    msk1 = nibabel.load(dir   + v.split('_flipped')[0] + '/AllLabels.nii.gz')
    msk2 = nibabel.load(dir_f + v + '/AllLabels_back2normal.nii.gz')

    msk3 = nibabel.Nifti1Image( msk1.get_data() + msk2.get_data() ,header=msk1.header , affine=msk1.affine)
    nibabel.save(msk3,dir + v.split('_flipped')[0] + '/AllLabels_Left_and_Right.nii.gz')