from keras.models import load_model
import sys
import os
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from otherFuncs.smallFuncs import listSubFolders
# sys.path.append(os.path.dirname(__file__))
from otherFuncs.datasets import percentageRandomDivide
from otherFuncs import params

Dir = params.directories.WhichExperiment.Dataset.address
TrainList, TestList = percentageRandomDivide(0.2, listSubFolders(Dir))

# os.copyfile()
