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

UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
UserInfoB['simulation'] = UserInfoB['simulation']()
K = smallFuncs.gpuSetting(str(UserInfoB['simulation'].GPU_Index))

def main_test(params, UserEntry):

    def running_main(UserInfoB):
        
        def Run(UserInfoB):
            params = paramFunc.Run(UserInfoB, terminal=True)
            Data, params = datasets.loadDataset(params)
            choosingModel.check_Run(params, Data)
            K.clear_session()

        def merge_results_and_apply_25D(UserInfoB):
            UserInfoB['best_network_MPlanar'] = True
            params = paramFunc.Run(UserInfoB, terminal=True)
            Output = params.WhichExperiment.Experiment.exp_address + '/results/' + params.WhichExperiment.Experiment.subexperiment_name
            os.system("mkdir {Output}/2.5D_MV")
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

        applyPreprocess.main(paramFunc.Run(UserInfoB, terminal=True), 'experiment')

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

    def run_Left(UserEntry):
        running_main(UserInfoB)
        for subj in params.directories.test.input.Subjects:
            smallFuncs.Save_AllNuclei_inOne(subj.address + '/left/2.5D_MV' , mode='')

    def run_Right(UserEntry):

        def flip_inputs():
            subjects = params.directories.test.input.Subjects.copy()
            subjects.update(params.directories.train.input.Subjects)

            for subj in subjects.values(): 
                os.system("cd %s;for n in *nii.gz; do fslswapdim $n -x y z $n; mv $n flipped_$n ; done"%(subj.address))   
            
        def unflip_inputs():

            subjects = params.directories.test.input.Subjects.copy()
            subjects.update(params.directories.train.input.Subjects)

            for subj in subjects.values():           
                os.system("cd %s;for n in  *.nii.gz right/*/*.nii.gz; do fslswapdim $n -x y z $n; done"%(subj.address)) 
                os.system("cd %s;for n in *.nii.gz ; do mv $n ${n#*_} ; done"%(subj.address))  # ${a#*_}   

        flip_inputs()
        running_main(UserInfoB)
        unflip_inputs()
        for subj in params.directories.test.input.Subjects:
            smallFuncs.Save_AllNuclei_inOne(subj.address + '/right/2.5D_MV' , mode='')          
        
    if UserInfoB['thalamic_side'].left:  
        run_Left(UserEntry)
    if UserInfoB['thalamic_side'].right: 
        run_Right(UserEntry)

    if UserInfoB['thalamic_side'].left and UserInfoB['thalamic_side'].right: 

        for subj in params.directories.test.input.Subjects:

            load_side = lambda side: nib.load(subj.address + '/' + side + '/2.5D_MV/AllLabels.nii.gz')
            left,right = load_side('left'), load_side('right')
            smallFuncs.saveImage(image= left.get_data() + right.get_data(), affine=left.affine , header=left.header , outDirectory=subj_address + '/left/AllLabels_Left_and_Right.nii.gz')
