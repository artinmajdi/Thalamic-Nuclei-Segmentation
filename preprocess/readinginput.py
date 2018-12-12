import nibabel as nib
import collections
from otherFuncs.smallFuncs import listSubFolders
from preprocess.normalizeInput import normalizeMain

# import os
# os.path.dirname('/array/ssd/msmajdi/experiments')

struct = collections.namedtuple('struct' , 'Image CropMask ThalamusMask Header Affine InputAddress')

def mainloadingImage(params , Input):

    im = nib.load(Input.Address + '/' + Input.Files.ImageProcessed + '.nii.gz')

    class InputImages:
        Image   = normalizeMain( params , im.get_data() )
        Label   = nib.load(Input.Files.label.Address + '/' + Input.Files.label.LabelProcessed + '.nii.gz').get_data()
        Header  = im.header
        Affine  = im.affine
        Address = Input.Address

    return InputImages
