import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess


# UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)


params = paramFunc.Run(UserInfo.__dict__, terminal=True)
K = smallFuncs.gpuSetting(params.WhichExperiment.HardParams.Machine.GPU_Index)



wait = input('press a key')

K.clear_session()
