import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess

UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
class InitValues:
    Nuclei_Indexes = UserInfoB['simulation'].nucleus_Index.copy()
    slicingDim     = UserInfoB['simulation'].slicingDim.copy()

print('slicingDim' , InitValues.slicingDim , 'Nuclei_Indexes' , InitValues.Nuclei_Indexes , 'GPU:  ', UserInfoB['simulation'].GPU_Index)

def gpuSetting(params):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
    return K

def Run(UserInfoB):

    def HierarchicalStages(UserInfoB):

        def HCascade_Parents_Identifier(InitValues):
            b = smallFuncs.NucleiIndex(1,'HCascade')
            list_HC = []
            for ix in b.child:
                c = smallFuncs.NucleiIndex(ix,'HCascade')
                if c.child and bool(set(InitValues.Nuclei_Indexes) & set(c.child)): list_HC.append(ix)
            return list_HC

        def remove_Thalamus_From_List(InitValues):
            Nuclei_Indexes = InitValues.Nuclei_Indexes.copy()
            if 1 in Nuclei_Indexes: Nuclei_Indexes.remove(1)
            return Nuclei_Indexes

        print('************ stage 1 ************')
        for UserInfoB['simulation'].nucleus_Index in [1]:
            Run_SingleNuclei(UserInfoB)

        print('************ stage 2 ************')               
        for UserInfoB['simulation'].nucleus_Index in HCascade_Parents_Identifier(InitValues):
            Run_SingleNuclei(UserInfoB)

        print('************ stage 3 ************')
        for UserInfoB['simulation'].nucleus_Index in remove_Thalamus_From_List(InitValues):
            Run_SingleNuclei(UserInfoB)

    def CacadeStages(UserInfoB):

        for UserInfoB['simulation'].nucleus_Index in InitValues.Nuclei_Indexes:
            Run_SingleNuclei(UserInfoB)

    def Run_SingleNuclei(UserInfoB):

        for sd in InitValues.slicingDim:

            print(' Nucleus:  ', UserInfoB['simulation'].nucleus_Index  , 'GPU:  ', UserInfoB['simulation'].GPU_Index , 'slicingDim',sd)
            UserInfoB['simulation'].slicingDim = [sd]

            params       = paramFunc.Run(UserInfoB)
            Data, params = datasets.loadDataset(params)
            
            choosingModel.check_Run(params, Data)

    MethodType = params.WhichExperiment.HardParams.Model.Method.Type
    if   MethodType == 'HCascade':  HierarchicalStages(UserInfoB)
    elif MethodType == 'Cascade':   CacadeStages(UserInfoB)
    elif MethodType == 'singleRun': Run_SingleNuclei(UserInfoB)

params = paramFunc.Run(UserInfoB)

datasets.movingFromDatasetToExperiments(params)
applyPreprocess.main(params, 'experiment')
K = gpuSetting(params)

Run(UserInfoB)

K.clear_session()
