
import nibabel as nib
import numpy as np
import sys, os


Link = True if '-l' in sys.argv else False
inputs = [os.getcwd() + '/' + x for x  in sys.argv[1:] if '-l' not in x]

# dir = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/pre-steps/CSFn/cropped_Image/step2_uncropped/vimp2_case1'
# Link = True
# inputs = [dir + '/crop_t1.nii.gz' , dir + '/Label/1-THALAMUS.nii.gz']

a = {}
for ix , dir in enumerate(inputs):
    
    a[ix] = nib.viewers.OrthoSlicer3D(nib.load(dir).get_data())

    if Link and ix > 0: a[0].link_to(a[ix])

a[0].show()








# dir = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/pre-steps/CSFn/cropped_Image/'
# a = '/array/ssd/msmajdi/experiments/keras/exp5_CSFn/crossVal/ET/a/vimp2_G_7T_ET/PProcessed.nii.gz' # step0_orig_crop/vimp2_case2/crop_t1.nii.gz'
# # b = 'step2_resliced_for_croppedInput/vimp2_case2/crop_t1.nii.gz'
# c = '/array/ssd/msmajdi/experiments/keras/exp5_CSFn/train/Main/vimp2_824_05212013_JS/PProcessed.nii.gz'
