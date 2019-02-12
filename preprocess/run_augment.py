import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from augmentA import main_augment
from otherFuncs import smallFuncs
from Parameters import paramFunc, UserInfo


params = paramFunc.Run(UserInfo.__dict__)

print('***********' , 'Nuclei:',params.WhichExperiment.Nucleus.name , '  GPU:',params.WhichExperiment.HardParams.Machine.GPU_Index , \
'  Epochs:', params.WhichExperiment.HardParams.Model.epochs,'  Dataset:',params.WhichExperiment.Dataset.name , \
'  Experiment: {',params.WhichExperiment.Experiment.name ,',', params.WhichExperiment.SubExperiment.name,'}')



main_augment( params , 'Linear', 'experiment')
params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
main_augment( params , 'NonLinear' , 'experiment')
