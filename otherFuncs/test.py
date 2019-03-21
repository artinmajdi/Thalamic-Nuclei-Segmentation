import numpy as np
import nibabel as nib
import os, sys
import skimage
import csv
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
params = paramFunc.Run(UserInfo.__dict__)
import pandas as pd
import h5py
import pickle
from tqdm import tqdm
import keras

subj = params.directories.Train.Input.Subjects['vimp2_A']

keras.utils.Sequence()
f = h5py.File(params.directories.Test.Result + '/Data.hdf5','r')


f['Train/Image'].shape
import numpy as np

np.arange(4,6)
