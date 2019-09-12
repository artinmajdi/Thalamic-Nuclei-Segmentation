
import nibabel as nib
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from skimage import measure


def show_middle_Slice():
    def imShow(*args):
        _, axes = plt.subplots(1,len(args))
        for ax, im in enumerate(args):
            axes[ax].imshow(im,cmap='gray')

        plt.show()

        return True

    def findMiddleSlice(dir_msk):
        obj = measure.regionprops(measure.label(nib.load(dir_msk).get_data()))
        return int((obj[0].bbox[-1] + obj[0].bbox[2])/2)


    for ix, en in enumerate(sys.argv):
        if en == '-image': dir_im  = os.getcwd() + '/' + sys.argv[ix+1]
        if en == '-mask':  dir_msk = os.getcwd() + '/' + sys.argv[ix+1]
        

    sc = findMiddleSlice(dir_msk)


    imm = np.squeeze(nib.load(dir_im).slicer[:,:,sc:sc+1].get_data())
    mskk = np.squeeze(nib.load(dir_msk).slicer[:,:,sc:sc+1].get_data())

    imm2 = imm/imm.max() + mskk*0.5

    imShow(imm2 , imm)


def showNibabel():
    Link = True if '-l' in sys.argv else False

    inputs = [ x for x  in sys.argv[1:] if '-l' not in x]

    print(inputs)
    for cnt, x in enumerate(inputs):
        if '/array' not in x:
            inputs[cnt] = os.getcwd() + '/' + inputs[cnt] 

    print(inputs)

    a = {}
    for ix , dir in enumerate(inputs):
        
        a[ix] = nib.viewers.OrthoSlicer3D(nib.load(dir).get_data())

        if Link and ix > 0: a[0].link_to(a[ix])

    a[0].show()


if ('-image' in sys.argv) and ('-mask' in sys.argv): show_middle_Slice()
else: showNibabel()





# dir = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/pre-steps/_Dataset1/cropped_Image/'
# a = '/array/ssd/msmajdi/experiments/keras/exp5_CSFn/crossVal/ET/a/vimp2_G_7T_ET/PProcessed.nii.gz' # step0_orig_crop/vimp2_case2/crop_t1.nii.gz'
# # b = 'step2_resliced_for_croppedInput/vimp2_case2/crop_t1.nii.gz'
# c = '/array/ssd/msmajdi/experiments/keras/exp5_CSFn/train/Main/vimp2_824_05212013_JS/PProcessed.nii.gz'
