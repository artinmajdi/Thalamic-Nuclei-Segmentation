import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
from preprocess import applyPreprocess
from Parameters import paramFunc, UserInfo

UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
params = paramFunc.Run(UserInfoB)
params.preprocess.Mode = True

applyPreprocess.main(params, 'experiment')

