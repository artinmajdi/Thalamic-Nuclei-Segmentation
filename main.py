import os
import sys

sys.path.append(os.path.dirname(__file__))
import otherFuncs.smallFuncs as smallFuncs
from otherFuncs import datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess
from keras import backend as K
import nibabel as nib
import numpy as np

UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
UserInfoB['simulation'] = UserInfoB['simulation']()
K = smallFuncs.gpuSetting(str(UserInfoB['simulation'].GPU_Index))

def main(UserInfoB):

    params = paramFunc.Run(UserInfoB, terminal=True)

    def Save_AllNuclei_inOne(Directory, mode='_PProcessed'):
        mask = []
        for cnt in (1,2,4,5,6,7,8,9,10,11,12,13,14):
            name = smallFuncs.Nuclei_Class(index=cnt).name
            dirr = Directory + '/' + name + mode + '.nii.gz'                                   
            if cnt == 1:
                thalamus_mask = nib.load( dirr )  

            else:
                if os.path.isfile(dirr):
                    msk = nib.load(dirr).get_data()  
                    if mask == []: 
                        mask = cnt*msk 
                    else:
                        mask_temp = mask.copy()
                        mask_temp[msk == 0] = 0
                        x = np.where(mask_temp > 0)
                        if x[0].shape[0] > 0: 
                            fg = np.random.randn(x[0].shape[0])
                            fg1, fg2 = fg >= 0 , fg < 0
                            mask[x[0][fg1],x[1][fg1],x[2][fg1]] = 0
                            msk[x[0][fg2],x[1][fg2],x[2][fg2]] = 0

                        mask += cnt*msk 
                    

        smallFuncs.saveImage( mask , thalamus_mask.affine , thalamus_mask.header, Directory + '/AllLabels.nii.gz')

    def running_main(UserInfoB):
        
        def Run(UserInfoB):
            params = paramFunc.Run(UserInfoB, terminal=True)

            
            print('\n',params.WhichExperiment.Nucleus.name , 'SD: ' + str(UserInfoB['simulation'].slicingDim) , 'GPU: ' + str(UserInfoB['simulation'].GPU_Index),'\n')
            Data, params = datasets.loadDataset(params)
            choosingModel.check_Run(params, Data)
            K.clear_session()

        def merge_results_and_apply_25D(UserInfoB):
            UserInfoB['best_network_MPlanar'] = True
            params = paramFunc.Run(UserInfoB, terminal=True)
            # Output = params.WhichExperiment.Experiment.exp_address + '/results/' + params.WhichExperiment.Experiment.subexperiment_name
            # os.system("mkdir {Output}/2.5D_MV")
            smallFuncs.apply_MajorityVoting(params)

        def predict_thalamus_for_sd0(UserI):
            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]
            UserI['simulation'].Use_Coronal_Thalamus_InSagittal = True
            Run(UserI)

            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            Run(UserI)

        def predict_multi_thalamus(UserI):
            UserI['simulation'].nucleus_Index = [1]
            Run(UserI)
            UserI['simulation'].nucleus_Index = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            Run(UserI)

        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40
        UserInfoB['simulation'].slicingDim = [0]
        UserInfoB['simulation'].nucleus_Index = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        predict_thalamus_for_sd0(UserInfoB)

        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30
        UserInfoB['simulation'].slicingDim = [1]
        UserInfoB['simulation'].nucleus_Index = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        predict_multi_thalamus(UserInfoB)

        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
        UserInfoB['simulation'].slicingDim = [2]
        UserInfoB['simulation'].nucleus_Index = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        UserInfoB['simulation'].Use_Coronal_Thalamus_InSagittal = False
        predict_multi_thalamus(UserInfoB)

        merge_results_and_apply_25D(UserInfoB)

    def run_Left(UserInfoB):
        UserInfoB['thalamic_side'].active_side = 'left'
        running_main(UserInfoB)

        params = paramFunc.Run(UserInfoB, terminal=True)
        for subj in params.directories.Test.Input.Subjects.values():
            Save_AllNuclei_inOne(subj.address + '/left/2.5D_MV' , mode='')

    def run_Right(UserInfoB):

        def flip_inputs():
            subjects = params.directories.Test.Input.Subjects.copy()
            subjects.update(params.directories.Train.Input.Subjects)

            for subj in subjects.values(): 
                os.system("cd %s;for n in *nii.gz; do fslswapdim $n -x y z $n; mv $n flipped_$n ; done"%(subj.address))   
            
        def unflip_inputs():

            subjects = params.directories.Test.Input.Subjects.copy()
            subjects.update(params.directories.Train.Input.Subjects)

            for subj in subjects.values():           
                os.system("cd %s;for n in  *.nii.gz right/*/*.nii.gz; do fslswapdim $n -x y z $n; done"%(subj.address)) 
                os.system("cd %s;for n in *.nii.gz ; do mv $n ${n#*_} ; done"%(subj.address))  # ${a#*_}   

        UserInfoB['thalamic_side'].active_side = 'right'
        flip_inputs()
        running_main(UserInfoB)
        unflip_inputs()
        for subj in params.directories.Test.Input.Subjects:
            Save_AllNuclei_inOne(subj.address + '/right/2.5D_MV' , mode='')          
        
    def merging_left_right_labels(UserInfoB):
        params = paramFunc.Run(UserInfoB, terminal=True)
        for subj in params.directories.Test.Input.Subjects:

            load_side = lambda side: nib.load(subj.address + '/' + side + '/2.5D_MV/AllLabels.nii.gz')
            left,right = load_side('left'),load_side('right')
            smallFuncs.saveImage(Image= left.get_data() + right.get_data(), Affine=left.affine , Header=left.header , outDirectory=subj.address + '/left/AllLabels_Left_and_Right.nii.gz')

    applyPreprocess.main(paramFunc.Run(UserInfoB, terminal=True))

    TS = UserInfoB['thalamic_side']
    if TS.left:              run_Left(UserInfoB)
    if TS.right:             run_Right(UserInfoB)
    if TS.left and TS.right: merging_left_right_labels(UserInfoB)


main(UserInfoB)