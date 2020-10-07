import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import nibabel as nib
import collections
from preprocess.normalizeA import main_normalize

struct = collections.namedtuple('struct' , 'Image CropMask ThalamusMask Header Affine InputAddress')

def mainloadingImage(params , Input):

    im = nib.load(Input.address + '/' + Input.Files.ImageProcessed + '.nii.gz')

    class InputImages:
        Image   = main_normalize( params , im.get_data() )
        Label   = nib.load(Input.Files.label.address + '/' + Input.Files.label.LabelProcessed + '.nii.gz').get_data()
        Header  = im.header
        Affine  = im.affine
        address = Input.address

    return InputImages
