import os, sys
sys.path.append(os.path.dirname(os.path.dirname('/array/ssd/msmajdi/code/thalamus/keras/otherFuncs/run_datasets.py')))
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo
import preprocess.applyPreprocess as applyPreprocess
import h5py
import numpy as np
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
mode = 'experiment'

#! reading the user input parameters via terminal
UserInfoB = UserInfo.__dict__ # smallFuncs.terminalEntries(UserInfo.__dict__)
UserInfoB['simulation'].slicingDim = [2]
UserInfoB['simulation'].nucleus_Index = [2]
params = paramFunc.Run(UserInfoB, terminal=True)
# params.WhichExperiment.Dataset.CreatingTheExperiment = True

#! copying the dataset into the experiment folder
datasets.movingFromDatasetToExperiments(params)

# params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
# params = smallFuncs.inputNamesCheck(params, mode)
# #! needs to run preprocess first to add the PPRocessed.nii.gz files

# #! preprocessing the data
# if params.preprocess.Mode: applyPreprocess.main(params, mode)
# params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)

# #! loading the dataset
Data, params = datasets.loadDataset(params)

obj = skimage.measure.regionprops( (Data.Train.Image > 0.5).astype(np.int32) )
ShapeSz = [obj[0].bbox[d+3]-obj[0].bbox[d] for d in range(3)]

a = 2**(params.WhichExperiment.HardParams.Model.num_Layers - 1)
new_inputSize = [  a * np.ceil(s / a) if s % a != 0 else s for s in ShapeSz ]


# a = skimage.transform.rescale( Data.Train.Image[:40,...] , (1,2,2,1), order=3)
# b = skimage.transform.rescale( Data.Train.Mask[:40,...]  , (1,2,2,1), order=1)

# a1 = nib.viewers.OrthoSlicer3D(a,title='original image')
# b1 = nib.viewers.OrthoSlicer3D(b,title='scaled')
# a1.link_to(b1)
# a1.show()

# a.shape
# obj = skimage.measure.regionprops( (a > 0.5).astype(np.int32) )
# obj[0].bbox


# Data.Train.Image.shape
# # aF = skimage.transform.rescale( Data.Train.Image , (1,2,2,1), order=3, clip=True)
# obj2 = skimage.measure.regionprops( (Data.Train.Image > 0.5).astype(np.int32) )
# obj2[0].bbox
















print('---')
