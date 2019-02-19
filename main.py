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
params = paramFunc.Run(UserInfoB)
NucleiIndexes = UserInfoB['nucleus_Index']

def Run_SingleNuclei(UserInfoB):
    
    def gpuSetting(params):
        os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
        import tensorflow as tf
        from keras import backend as K
        K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
        return K
        
    params = paramFunc.Run(UserInfoB)
    Data, params = datasets.loadDataset(params)
    K = gpuSetting(params)
    choosingModel.check_Run(params, Data)

    return K

datasets.movingFromDatasetToExperiments(params)

applyPreprocess.main(params, 'experiment')
params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)

for UserInfoB['nucleus_Index'] in NucleiIndexes:  K = Run_SingleNuclei(UserInfoB)

K.clear_session()