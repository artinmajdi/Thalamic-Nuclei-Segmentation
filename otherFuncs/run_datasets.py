import os, sys
sys.path.append(os.path.dirname(os.path.dirname('/array/ssd/msmajdi/code/thalamus/keras/otherFuncs/run_datasets.py')))
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo
import preprocess.applyPreprocess as applyPreprocess
import h5py
import numpy as np

mode = 'experiment'

#! reading the user input parameters via terminal
UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)

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
FullData = Data.__dict__

f.close()
f = h5py.File(params.WhichExperiment.Experiment.address + '/data.h5py' , 'w')
for mode in list(FullData):
    # print(mode)
    # g = f.create_group(mode)
    DataMode = FullData[mode]

    if mode == 'Train' or mode == 'Validation':
        for imMsk in DataMode.__dict__.items():
            f.create_dataset(name='%s/%s'%(mode,imMsk[0]),data=imMsk[1])

# else:
#     for subject in DataMode:
#         DataSubject = DataMode[subject].__dict__
#         for tag in list(DataSubject):
#             print(DataSubject[tag])
#         break

dtst = f['Train/Image']
dtst[:100,...].shape
Image = np.array(f['Train/Image'])
Mask = np.array(f['Train/Mask'])

np.savez
f['Train/Image']
n[:,:100,:,:].shape

mode = list(Datadt)[0]
subData = Datadt[mode]
for subject in subData:
    Tags = list(subData[subject].__dict__)
    print(Tags)
    break


    # mode = aa[ix][0]
# aa[3]
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
