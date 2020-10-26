
import nibabel as nib
# import numpy as np
import sys, os
# import matplotlib.pyplot as plt
# from skimage import measure


def mkDir(Dir):
    if not os.path.isdir(Dir): os.makedirs(Dir)
    return Dir

def saveImage(Image , Affine , Header , outDirectory):
    mkDir(outDirectory.split(os.path.basename(outDirectory))[0])
    out = nib.Nifti1Image((Image).astype('float32'),Affine)
    out.get_header = Header
    nib.save(out , outDirectory)



for ix, en in enumerate(sys.argv):
    if en == '-i': dir_in  = os.getcwd() + '/' + sys.argv[ix+1]
    if en == '-o': dir_out = os.getcwd() + '/' + sys.argv[ix+1]
        

# dir_in = '/array/ssd/msmajdi/experiments/keras/exp4/test/Main/vimp2_case17_CSFn/PProcessed.nii.gz'
# dir_out = '/array/ssd/msmajdi/experiments/keras/exp4/test/Main/vimp2_case17_CSFn/PProcessed2.nii.gz'

imF = nib.load(dir_in)
im = imF.get_fdata()
im2 = 1 - im / im.max()


saveImage(im2 , imF.affine , imF.header , dir_out)





