import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import otherFuncs.smallFuncs as smallFuncs
import preprocess.applyPreprocess as applyPreprocess
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo

# UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
params = paramFunc.Run(UserInfo.__dict__, terminal=True)
params.preprocess.Mode = True

applyPreprocess.main(params, 'experiment')

