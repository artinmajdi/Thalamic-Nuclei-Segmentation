import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras_run/')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocess.augmentA import main_augment
import otherFuncs.smallFuncs as smallFuncs
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo


params = paramFunc.Run(UserInfo.__dict__)
params.Augment.Mode = True

print('***********' , 'Nuclei:',params.WhichExperiment.Nucleus.name , '  GPU:',params.WhichExperiment.HardParams.Machine.GPU_Index , \
'  Epochs:', params.WhichExperiment.HardParams.Model.epochs,'  Dataset:',params.WhichExperiment.Dataset.name , \
'  Experiment: {',params.WhichExperiment.Experiment.name ,',', params.WhichExperiment.SubExperiment.name,'}')



main_augment( params , 'Linear', 'experiment')
params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
main_augment( params , 'NonLinear' , 'experiment')
