from keras.models import load_model
import sys
import os
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from otherFuncs.smallFuncs import listSubFolders
# sys.path.append(os.path.dirname(__file__))
from otherFuncs.datasets import percentageRandomDivide
from otherFuncs import params
from shutil import copytree


# Dir_Prior = next(os.walk(params.directories.WhichExperiment.Dataset.address))[1]
# len(Dir_Prior)

# List = listSubFolders(params.directories.WhichExperiment.Dataset.address)
# TestParams = params.directories.WhichExperiment.Dataset.Test
# _, TestList = percentageRandomDivide(TestParams.percentage, List) if 'percentage' in TestParams.mode else TestParams.subjects
# for subjects in List:
#     DirOut = params.directories.Test.address if subjects in TestList else params.directories.Train.address
#     copytree(params.directories.WhichExperiment.Dataset.address + '/' + subjects, DirOut + '/' + subjects )
