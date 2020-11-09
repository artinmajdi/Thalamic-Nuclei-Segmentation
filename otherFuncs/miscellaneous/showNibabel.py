
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
        obj = measure.regionprops(measure.label(nib.load(dir_msk).get_fdata()))
        return int((obj[0].bbox[-1] + obj[0].bbox[2])/2)


    for ix, en in enumerate(sys.argv):
        if en == '-image': dir_im  = os.getcwd() + '/' + sys.argv[ix+1]
        if en == '-mask':  dir_msk = os.getcwd() + '/' + sys.argv[ix+1]
        

    sc = findMiddleSlice(dir_msk)


    imm = np.squeeze(nib.load(dir_im).slicer[:,:,sc:sc+1].get_fdata())
    mskk = np.squeeze(nib.load(dir_msk).slicer[:,:,sc:sc+1].get_fdata())

    imm2 = imm/imm.max() + mskk*0.5

    imShow(imm2 , imm)


def showNibabel():
    Link = True if '-l' in sys.argv else False

    inputs = [ x for x  in sys.argv[1:] if '-l' not in x]

    print(inputs)
    for cnt, x in enumerate(inputs):
        if not '/' == x[0]:
            inputs[cnt] = os.getcwd() + '/' + inputs[cnt] 

    print(inputs)

    a = {}
    for ix , dir in enumerate(inputs):
        
        a[ix] = nib.viewers.OrthoSlicer3D(nib.load(dir).get_fdata())

        if Link and ix > 0: a[0].link_to(a[ix])

    a[0].show()


if ('-image' in sys.argv) and ('-mask' in sys.argv): show_middle_Slice()
else: showNibabel()

