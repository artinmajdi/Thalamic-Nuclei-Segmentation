import nibabel as nib
import collections
from smallCodes import listSubFolders

struct = collections.namedtuple('struct' , 'Image CropMask ThalamusMask TestAddress')

def loadingImage(NucleusName , dir):

    im = nib.load(dir + '/WMnMPRAGE_bias_corr.nii.gz')
    CropMask = nib.load(dir + '/MyCrop2_Gap20.nii.gz')

    if '1-THALAMUS' not in NucleusName:
        ThalamusMask = nib.load(dir + '/MyCrop2_Gap20.nii.gz')
    else:
        ThalamusMask = []

    return struct(im , CropMask , ThalamusMask,dir)

def mainloadingImage(params):

    FullData = list()
    if params.MultipleTest == 0:
        structList = loadingImage(params.NucleusName , params.Directory_input)
        FullData.append(structList)
    else:
        subFolders = listSubFolders(params.Directory_input)
        for sFi in range(len(subFolders)):
            structList = loadingImage(params.NucleusName , params.Directory_input + '/' + subFolders[sFi])
            FullData.append(structList)

    return FullData
