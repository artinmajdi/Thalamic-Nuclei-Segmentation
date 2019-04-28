import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append( os.path.dirname(os.path.dirname(__file__)) )
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc


params = paramFunc.Run(UserInfo.__dict__, terminal=True)
print(params.WhichExperiment.SubExperiment.name)
