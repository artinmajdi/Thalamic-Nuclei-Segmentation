import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
from preprocess import applyPreprocess
from Parameters import paramFunc, UserInfo

params = paramFunc.Run(UserInfo.__dict__)

#! mode: 1: on train & test folders in the experiment
#! mode: 2: on individual image
params = smallFuncs.terminalEntries(params)
applyPreprocess.main(params, 'experiment')

