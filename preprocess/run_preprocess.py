import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import params, smallFuncs
from preprocess.preprocessA import main_preprocess

#! mode: 1: on train & test folders in the experiment
#! mode: 2: on individual image
params = smallFuncs.terminalEntries(params)
main_preprocess(params, 'experiment')