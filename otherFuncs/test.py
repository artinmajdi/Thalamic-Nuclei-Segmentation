import numpy as np
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append( os.path.dirname(os.path.dirname(__file__)) )
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import keras
import h5py
import nibabel as nib

params = paramFunc.Run(UserInfo.__dict__, terminal=True)
K = smallFuncs.gpuSetting(params.WhichExperiment.HardParams.Machine.GPU_Index)

# model = keras.models.load_model(params.directories.Train.Model + '/model.h5')
# subject = '/vimp2_869_06142013_BL_MS/'
subject = '/vimp2_915_07112013_LC_MS/'
dir = '/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/test/Main' + subject
image = nib.load(dir + 'PProcessed.nii.gz').get_data()
mask = nib.load(dir + 'Label/1-THALAMUS_PProcessed.nii.gz').get_data()

dir2 = '/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/results/sE11_HCascade_wRot7_6cnts_sd1_Dt0.3_LR0.001_NL3_FM64_WoET_Init_From_3T_AutoDim_BestEpch' + subject
pred = nib.load(dir2 + '1-THALAMUS.nii.gz').get_data()


def imshow(mask,pred,image):
    a = nib.viewers.OrthoSlicer3D(image + 500*mask, title='orig')
    b = nib.viewers.OrthoSlicer3D(image + 500*pred, title='pred')
    a.link_to(b)
    a.show()

imshow(mask,pred,image)

print(params.directories.Train)


# dir = '/home/artinl/Documents/research/'
# listS = [s for s in os.listdir(dir) if 'sE' in s]

# ind = 0
# print(listS[ind])
# with open(dir + listS[0] + '/1-THALAMUS/UserInfo.pkl','rb') as f:
#     d = pickle.load(f)

# print(d['simulation'].Multiply_By_Thalmaus)


# print('---')
