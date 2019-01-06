
import os, sys
__file__ = '/array/ssd/msmajdi/code/thalamus/keras'  #! only if I'm using Hydrogen Atom
sys.path.append(__file__)
import numpy as np
from tensorflow import math


a = np.random.random(10)
b = math.multiply(a,a)

math.reduce
print('--')


# import h5py
# im = nib.load('/array/ssd/msmajdi/experiments/keras/exp1_tmp/train/vimp2_2309_07072016/WMnMPRAGE_bias_corr.nii.gz').get_data()
# f = h5py.File('/array/ssd/msmajdi/experiments/keras/exp1_tmp/train/Train.hdf5','w')
# f.create_dataset('images',im,dtype='i')

# import h5py
# Create a new file using defaut properties.
#
# file = h5py.File('dset.h5','w')
# dataset = file.create_dataset("dset",im[:,0,0], h5py.h5t.STD_I32BE)
# print ("Dataset dataspace is", dataset.shape)
# print ("Dataset Numpy datatype is", dataset.dtype)
# print ("Dataset name is", dataset.name)
# print ("Dataset is a member of the group", dataset.parent)
# print ("Dataset was created in the file", dataset.file)
# # Close the file before exiting
# file.close()


print('-')