import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os, sys
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
from scipy import ndimage


dir = '/media/data1/artin/other/dataBackupServer/20priors/vimp2_943_07242013_PA//'
im = nib.load(dir + 'Manual_Delineation_Sanitized/1-THALAMUS.nii.gz')

Names = dict({1:['posterior',[8,10,11]], 2:['lateral',[4,5,6,7]] , 3:['2-AV',[2]] , 4:['12-MD-Pf',[12]] })

for mIx in [1,2]:
    for cnt, nix in enumerate(Names[mIx][1]):
        name, _, _ = smallFuncs.NucleiSelection(ind=nix,organ='THALAMUS')
        msk = nib.load(dir + 'Manual_Delineation_Sanitized/' + name + '.nii.gz').get_data()
        Mask = msk if cnt == 0 else Mask + msk

    smallFuncs.saveImage(Mask > 0 , im.affine , im.header, dir + '/Manual_Delineation_Sanitized/' + Names[mIx][0] + '.nii.gz')


def closeMask(mask):
    struc = ndimage.generate_binary_structure(3,2)
    return ndimage.binary_closing(mask, structure=struc)
    
for cnt in range(1,5):
    msk = nib.load(dir + 'Manual_Delineation_Sanitized/' + Names[cnt][0] + '.nii.gz').get_data()
    Mask = msk if cnt == 1 else Mask + cnt*msk
    MaskClosed = closeMask(msk) if cnt == 1 else MaskClosed + cnt*closeMask(msk)

smallFuncs.saveImage(Mask , im.affine , im.header, dir + '/Manual_Delineation_Sanitized/All_4MainNuclei.nii.gz')
smallFuncs.saveImage(MaskClosed , im.affine , im.header, dir + '/Manual_Delineation_Sanitized/All_4MainNuclei_ImClosed.nii.gz')

# myshow(137,im, msk8,msk10,msk11,postriorMask)
