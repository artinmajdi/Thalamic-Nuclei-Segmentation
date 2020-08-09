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

UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
K = smallFuncs.gpuSetting(str(UserInfoB['simulation'].GPU_Index))


def running_main(UserInfoB):

    def Run(UserInfoB):
        params = paramFunc.Run(UserInfoB, terminal=True)
        Data, params = datasets.loadDataset(params)
        choosingModel.check_Run(params, Data)
        K.clear_session()

    def merge_results_and_apply_25D(UserInfoB):
        UserInfoB['best_network_MPlanar'] = True
        params = paramFunc.Run(UserInfoB, terminal=True)
        DT = params.WhichExperiment.Experiment.address + '/results'

        subEx_name = params.WhichExperiment.SubExperiment.name
        Output = DT + '/' + subEx_name

        os.system("mkdir {Output}; cd {Output}; mkdir sd0 sd1 sd2")

        for FM in [('_FM40', '/sd0'), ('_FM30', '/sd1'), ('_FM20', '/sd2')]:
            input = DT + '/' + subEx_name.replace('_FM00', FM[0]) + FM[1]
            os.system("cp -r %s/vimp* %s/" % (input, Output + FM[1]))

        smallFuncs.apply_MajorityVoting(params)

    def predict_thalamus_for_sd0(UserI):
        UserI['simulation'].slicingDim = [2]
        UserI['simulation'].nucleus_Index = [1]
        UserI['simulation'].Use_Coronal_Thalamus_InSagittal = True
        Run(UserI)

        UserI['simulation'].slicingDim = [0]
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
    Run(UserInfoB)

    UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
    UserInfoB['simulation'].slicingDim = [2]
    UserInfoB['simulation'].nucleus_Index = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    UserInfoB['simulation'].Use_Coronal_Thalamus_InSagittal = False
    Run(UserInfoB)

    merge_results_and_apply_25D(UserInfoB)


if UserInfoB['wmn_csfn'] == 'csfn':
    UserInfoB['TypeExperiment'] = 8
elif UserInfoB['wmn_csfn'] == 'wmn':
    UserInfoB['TypeExperiment'] = 15

running_main(UserInfoB)
