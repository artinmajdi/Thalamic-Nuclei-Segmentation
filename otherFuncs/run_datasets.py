import os, sys
sys.path.append(os.path.dirname(os.path.dirname('/array/ssd/msmajdi/code/thalamus/keras_run/otherFuncs/run_datasets.py')))
from otherFuncs import smallFuncs, datasets
from Parameters import paramFunc
from Parameters import UserInfo
from preprocess import applyPreprocess
import h5py
import numpy as np
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
mode = 'experiment'

#! reading the user input parameters via terminal
UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
UserInfoB['slicingDim'] = [0]
UserInfoB['nucleus_Index'] = [1]
params = paramFunc.Run(UserInfoB)
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
subjects = list(Data.Train_ForTest)
BBOX = np.zeros((len(subjects),6))
Shape = np.zeros((len(subjects),3))
for ind in range(len(subjects)):
    data = Data.Train_ForTest[subjects[ind]]
    # Data.Train.Image.shape
    # data.Image.shape

    a = np.squeeze(data.Mask[...,0]).astype(np.int32)
    obj = skimage.measure.regionprops(a)
    BBOX[ind,...] = list(obj[0].bbox)
    Shape[ind,...] = list( data.Image.shape[:3] )

b = np.zeros((30,4))
b[:,:2] = BBOX[:,[0,3]]
b[:,2] = Shape[:,0]
b[:,3] = b[:,2]/2 - b[:,1] + 5

b
tuple( BBOX[:,:3].min(axis=0)  )  + tuple( BBOX[:,3:].max(axis=0) )





plt.plot(BBOX[:,0])

def myView(data):

    b = nib.viewers.OrthoSlicer3D(data.Mask)
    a = nib.viewers.OrthoSlicer3D(data.Image)
    a.link_to(b)
    a.show()

myView(data)

print('--')
# params.directories.Tr
# def saveHdf5(Data):
#     with h5py.File(params.WhichExperiment.Experiment.address + '/7T_wAug.h5py' , 'w') as f:
#         for subject in Data.Test:
#             f.create_dataset('Test/%s/Image'%(subject),data=Data.Test[subject].Image)
#             f.create_dataset('Test/%s/Mask'%(subject),data=Data.Test[subject].Mask)

#         for subject in Data.Train_ForTest:
#             f.create_dataset('Train/%s/Image'%(subject),data=Data.Train_ForTest[subject].Image)
#             f.create_dataset('Train/%s/Mask'%(subject),data=Data.Train_ForTest[subject].Mask)

#         f.visit(print)

# saveHdf5(Data)

# with h5py.File(params.WhichExperiment.Experiment.address + '/7T_wAug.h5py' , 'r') as g:
#     for subject in Data.Train_ForTest:
#         g.create_dataset('Test/%s/Image'%(subject + '_b'),data=Data.Test[subject].Image)
#         g.create_dataset('Test/%s/Mask'%(subject + '_b'),data=Data.Test[subject].Mask)
#     b = g['Train']['vimp2_A']['Image']
