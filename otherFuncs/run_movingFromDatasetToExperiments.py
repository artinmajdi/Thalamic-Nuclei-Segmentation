import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import params, datasets

datasets.movingFromDatasetToExperiments(params)
