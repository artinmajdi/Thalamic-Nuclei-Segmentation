import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/')
import otherFuncs.smallFuncs as smallFuncs
import preprocess.applyPreprocess as applyPreprocess
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo

params = paramFunc.Run(UserInfo.__dict__, terminal=True)

applyPreprocess.main(params)

