import nibabel as nib
import collections
from smallCodes import listSubFolders
from normalizeInput import normalizeMain

struct = collections.namedtuple('struct' , 'Image CropMask ThalamusMask Header Affine InputAddress')

def mainloadingImage(params , Input):
    
    im = nib.load(Input.Address + '/' + Input.Files.Cropped + '.nii.gz')

    class InputImages:
        Image   = normalizeMain( params , im.get_data() )
        Label   = nib.load(Input.Files.Nucleus.Address + '/' + Input.Files.Nucleus.Cropped + '.nii.gz').get_data()
        Header  = im.header 
        Affine  = im.affine 
        Address = Input.Address

    return InputImages

