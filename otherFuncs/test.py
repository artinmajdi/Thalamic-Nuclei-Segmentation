import nibabel as nib
import smallFuncs
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm 


dir_predictions = '/array/ssd/msmajdi/experiments/keras/exp6_uncropped/results/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CVs_all/sd2/'

for mode in ['/ET/' , '/Main/']:
    main_directory = '/array/ssd/msmajdi/experiments/keras/exp6_uncropped/crossVal'+mode # c/vimp2_988_08302013_CB/PProcessed.nii.gz'

    for cv in tqdm(['a/', 'b/' , 'c/' , 'd/' , 'e/', 'f/']):
        
        dir_predictions = '/array/ssd/msmajdi/experiments/keras/exp6_uncropped/results/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_' + cv + 'sd2/'

        if not os.path.exists(main_directory + cv): continue

        subjects = [s for s in os.listdir(main_directory + cv) if 'vimp' in s]
        for subj in tqdm(subjects):

            im = nib.load(main_directory + cv + subj + '/PProcessed.nii.gz').get_data()
            label = nib.load(main_directory + cv + subj + '/Label/1-THALAMUS_PProcessed.nii.gz').get_data()
            label = smallFuncs.fixMaskMinMax(label,subj)
            OP = nib.load(dir_predictions + subj + '/1-THALAMUS.nii.gz')
            original_prediction = OP.get_data()


            objects = measure.regionprops(measure.label(original_prediction))

            L = len(original_prediction.shape)
            if len(objects) > 1:
                area = []
                for obj in objects: area = np.append(area, obj.area)

                Ix = np.argsort(area)
                obj = objects[ Ix[-1] ]

                fitlered_prediction = np.zeros(original_prediction.shape)
                for cds in obj.coords:
                    fitlered_prediction[tuple(cds)] = True

                Dice = np.zeros(2)
                Dice[0], Dice[1] = 1, smallFuncs.mDice(fitlered_prediction > 0.5 , label > 0.5) 
                # np.savetxt(dir_predictions + subj + '/Dice_1-THALAMUS.txt' ,Dice,fmt='%1.4f')

                smallFuncs.saveImage(fitlered_prediction , OP.affine , OP.header , dir_predictions + subj + '/1-THALAMUS_biggest_obj.nii.gz')
        
        else:
            image = objects[0].image




print('---')


# a = nib.viewers.OrthoSlicer3D(fitlered_prediction)

# b = nib.viewers.OrthoSlicer3D(im)
# b.link_to(a)
# b.show()
