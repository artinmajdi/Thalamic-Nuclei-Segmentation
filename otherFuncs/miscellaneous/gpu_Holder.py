import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import os
import otherFuncs.smallFuncs as smallFuncs
import sys

params = paramFunc.Run(UserInfo.__dict__, terminal=True)
K = smallFuncs.gpuSetting(params.WhichExperiment.HardParams.Machine.GPU_Index)

wait = input('press a key')

K.clear_session()
