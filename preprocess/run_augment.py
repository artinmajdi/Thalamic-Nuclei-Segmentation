import os, sys
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras_run/')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import preprocess.augmentA as augmentA
import otherFuncs.smallFuncs as smallFuncs
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo

# UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
params = paramFunc.Run(UserInfo.__dict__, terminal=True)
params.Augment.Mode = True


augmentA.main_augment( params , 'Linear', 'experiment')
params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
augmentA.main_augment( params , 'NonLinear' , 'experiment')
