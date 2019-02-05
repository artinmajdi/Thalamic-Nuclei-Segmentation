import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import numpy as np
from Parameters import UserInfo, paramFunc
from otherFuncs import datasets, smallFuncs


params = paramFunc.Run(UserInfo.__dict__)
datasets.movingFromDatasetToExperiments(params)

params = smallFuncs.inputNamesCheck(params, 'experiment')
