import numpy as np
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append( os.path.dirname(os.path.dirname(__file__)) )
import otherFuncs.smallFuncs as smallFuncs
# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# import keras
# import h5py
# import nibabel as nib

# params = paramFunc.Run(UserInfo.__dict__, terminal=True)
# K = smallFuncs.gpuSetting(params.WhichExperiment.HardParams.Machine.GPU_Index)

smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Indexes
smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Names
print('--')
