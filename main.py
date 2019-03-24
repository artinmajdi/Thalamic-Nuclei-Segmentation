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
NucleiIndexes = UserInfoB['simulation'].nucleus_Index
slicingDim    = UserInfoB['simulation'].slicingDim
print('slicingDim' , slicingDim)

def gpuSetting(params):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
    return K

def Run(UserInfoB):

    def HierarchicalStages(UserInfoB):

        # stage 1
        print('************ stage 1 ************')
        UserInfoB['simulation'].nucleus_Index = [1]
        UserInfoB['gapDilation'] = 5
        Run_SingleNuclei(UserInfoB)

        # stage 2
        print('************ stage 2 ************')
        UserInfoB['gapDilation'] = 3
        for UserInfoB['simulation'].nucleus_Index in [1.1 , 1.2 , 1.3]:
            name,_,_ = smallFuncs.NucleiSelection(ind=UserInfoB['simulation'].nucleus_Index)
            print('      ', name , 'gpu: ',UserInfoB['simulation'].GPU_Index)
            Run_SingleNuclei(UserInfoB)

        print('************ stage 3 ************')
        # stage 3 ; final for now
        print('index',NucleiIndexes)
        _,FullIndexes ,_ = smallFuncs.NucleiSelection(ind=1)
        for UserInfoB['simulation'].nucleus_Index in FullIndexes[1:]:
            try:
                name,_,_ = smallFuncs.NucleiSelection(ind=UserInfoB['simulation'].nucleus_Index)
                print('      ', name , 'gpu: ',UserInfoB['simulation'].GPU_Index)
                Run_SingleNuclei(UserInfoB)
            except:
                print('failed')

    def CacadeStages(UserInfoB):

        for UserInfoB['simulation'].nucleus_Index in NucleiIndexes:
            print('    Nucleus:  ', UserInfoB['simulation'].nucleus_Index  , 'GPU:  ', UserInfoB['simulation'].GPU_Index)
            Run_SingleNuclei(UserInfoB)

    def Run_SingleNuclei(UserInfoB):

        for sd in slicingDim:
            UserInfoB['simulation'].slicingDim = [sd]
            params = paramFunc.Run(UserInfoB)
            Data, params = datasets.loadDataset(params)
            choosingModel.check_Run(params, Data)

    if params.WhichExperiment.HardParams.Model.Method.Type == 'HCascade': HierarchicalStages(UserInfoB)
    elif params.WhichExperiment.HardParams.Model.Method.Type == 'Cascade': CacadeStages(UserInfoB)
    elif params.WhichExperiment.HardParams.Model.Method.Type == 'singleRun': Run_SingleNuclei(UserInfoB)

def Finding_InputDimension(UserInfoB):
    UserInfoC = UserInfoB.copy()

    UserInfoC['readAugments'].Mode = True
    UserInfoC['ReadTrain'].SRI = True
    UserInfoC['ReadTrain'].ET = True
    UserInfoC['ReadTrain'].Main = True
    UserInfoC['InputPadding'].Automatic = True
    paramsB = paramFunc.Run(UserInfoC)
    paramsB = datasets.preAnalysis(paramsB)


    UserInfoB['InputPadding'].Automatic = False
    UserInfoB['InputPadding'].HardDimensions = paramsB.WhichExperiment.HardParams.Model.InputDimensions
    return UserInfoB


UserInfoB = Finding_InputDimension(UserInfoB)
params = paramFunc.Run(UserInfoB)

datasets.movingFromDatasetToExperiments(params)
applyPreprocess.main(params, 'experiment')
K = gpuSetting(params)

Run(UserInfoB)

K.clear_session()
