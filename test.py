import os, sys
__file__ = '/array/ssd/msmajdi/code/thalamus/keras'  #! only if I'm using Hydrogen Atom
sys.path.append(__file__)
from otherFuncs import params

subF = next(os.walk(params.WhichExperiment.Dataset.address))
os.listdir(params.WhichExperiment.Dataset.address)
