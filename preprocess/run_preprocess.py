import os, sys
sys.path.append('/array/ssd/msmajdi/code/CNN/')
import otherFuncs.smallFuncs as smallFuncs
import preprocess.applyPreprocess as applyPreprocess
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo

UserInfo = smallFuncs.terminalEntries(UserInfo.__dict__)
UserInfo['simulation'] = UserInfo['simulation']()
params = paramFunc.Run(UserInfo, terminal=True)

applyPreprocess.main(params)

