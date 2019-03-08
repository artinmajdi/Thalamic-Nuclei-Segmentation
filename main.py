import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from otherFuncs import smallFuncs, datasets
from modelFuncs import choosingModel
from Parameters import UserInfo, paramFunc
from preprocess import applyPreprocess

# TODO:  add a fixed seed number for random numbers
# TODO:  write the name of test and train subjects in model and results and dataset  to have it for the future
# TODO : look for a way to see epoch inside my loss function and use BCE initially and tyhen add Dice for higher epochs
UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
NucleiIndexes = UserInfoB['nucleus_Index']

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
        UserInfoB['nucleus_Index'] = [1]
        UserInfoB['gapDilation'] = 5
        Run_SingleNuclei(UserInfoB)

        # stage 2
        print('************ stage 2 ************')
        UserInfoB['gapDilation'] = 3
        for UserInfoB['nucleus_Index'] in [1.1 , 1.2, 1.3]:
            name,_,_ = smallFuncs.NucleiSelection(ind=UserInfoB['nucleus_Index'],organ='THALAMUS')
            print('      ', name , 'gpu: ',UserInfoB['GPU_Index'])
            Run_SingleNuclei(UserInfoB)

        print('************ stage 3 ************')
        # stage 3 ; final for now
        print('index',NucleiIndexes)
        for UserInfoB['nucleus_Index'] in NucleiIndexes[1:]:
            name,_,_ = smallFuncs.NucleiSelection(ind=UserInfoB['nucleus_Index'],organ='THALAMUS')
            print('      ', name , 'gpu: ',UserInfoB['GPU_Index'])            
            Run_SingleNuclei(UserInfoB)

    def CacadeStages(UserInfoB):

        for UserInfoB['nucleus_Index'] in NucleiIndexes:
            print('    Nucleus:  ', UserInfoB['nucleus_Index']  , 'GPU:  ', UserInfoB['GPU_Index'])
            Run_SingleNuclei(UserInfoB)

    def Run_SingleNuclei(UserInfoB):

        params = paramFunc.Run(UserInfoB)
        Data, params = datasets.loadDataset(params)
        choosingModel.check_Run(params, Data)

    if params.WhichExperiment.HardParams.Model.Method.Type == 'Hierarchical_Cascade': HierarchicalStages(UserInfoB)
    elif params.WhichExperiment.HardParams.Model.Method.Type == 'Cascade': CacadeStages(UserInfoB)


params = paramFunc.Run(UserInfoB)

datasets.movingFromDatasetToExperiments(params)
applyPreprocess.main(params, 'experiment')

K = gpuSetting(params)
Run(UserInfoB)

K.clear_session()
