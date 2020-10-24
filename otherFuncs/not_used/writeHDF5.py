import numpy as np
import nibabel as nib
import os, sys
import csv
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
params = paramFunc.Run(UserInfo.__dict__, terminal=True)
import pandas as pd
import h5py
import pickle
from tqdm import tqdm


def saving_OriginalDataset_AsHDF5(Input):

    #! Input should be either params or address to the experiment
    def Write_HDF5(f):
        def LoopOverAllSubjects(h):

            def saveSubject(k):
                def saveSubjectMask(kl):
                    allFiles = next(os.walk(kl.attrs['address']))[2]
                    Masks = [s for s in allFiles if '_PProcessed' in s]
                    for msk in Masks:
                        kl.create_dataset(name=msk.split('_PProcessed')[0],data=nib.load( kl.attrs['address'] + '/' + msk).get_fdata() )
                    return kl

                allFiles = next(os.walk(k.attrs['address']))[2]
                Image = [s for s in allFiles if 'PProcessed' in s]
                if len(Image) == 1:
                    imF = nib.load( k.attrs['address'] + '/' + Image[0])
                    dim = params.WhichExperiment.HardParams.Model.Method.InputImage2Dvs3D
                    k.create_dataset(name='imData',data=imF.get_fdata() ,chunks=tuple(imF.shape[:dim]) + (1,) ) # compression='gzip',
                    k.attrs['name'] = Image[0]

                    k.create_group('Label').attrs['address'] = k.attrs['address'] + '/Label'
                    saveSubjectMask(k['Label'])
                else:
                    print('more than one preprocessed image in the folder')
                return k

            subjects = [s for s in os.listdir(h.attrs['address']) if 'case' in s]
            for subj in tqdm(subjects[:3]):
                j = h.create_group(subj)
                j.attrs['address'] = h.attrs['address']  + '/' + subj
                saveSubject(j)

        def readTrainData():

            mode = 'train'
            def readAugmentFolders():
                f[mode].create_group('Augments').attrs['address'] = f[mode].attrs['address'] + '/Augments'
                AugList = [s for s in next(os.walk(f['train'].attrs['address'] + '/Augments'))[1] if 'w' in s]
                for subAug in AugList:
                    print(mode,'Augments',subAug)
                    f['train/Augments'].create_group(subAug).attrs['address'] = f['train/Augments'].attrs['address'] + '/' + subAug
                    LoopOverAllSubjects(f['train/Augments/%s'%(subAug)])

            def readNotAugmentFolders():

                f.create_group(mode).attrs['address'] = f.attrs['address']  + '/' + mode
                for subTrainFd in ['Main','SRI']:
                    print(mode,subTrainFd)
                    f[mode].create_group(subTrainFd).attrs['address'] = f[mode].attrs['address'] + '/' + subTrainFd
                    LoopOverAllSubjects(f['%s/%s'%(mode,subTrainFd)])

            readNotAugmentFolders()
            readAugmentFolders()

        def readTestData():
            mode = 'test'
            print(mode)
            f.create_group(mode).attrs['address'] = f.attrs['address']  + '/' + mode
            LoopOverAllSubjects(f[mode])

        readTrainData()
        readTestData()

    Directory = Input if isinstance(Input,str) else params.WhichExperiment.Experiment.address
    with h5py.File( Directory  + '/Data.h5py' , 'w') as f:
        f.attrs['address'] = Directory
        Write_HDF5(f)

    # with h5py.File( Directory + '/Data.h5py','r') as f:
    #     f['train/Main'].visit(print)
