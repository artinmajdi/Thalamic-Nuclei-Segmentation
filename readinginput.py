import nibabel as nib
import collections
from smallCodes import listSubFolders

struct = collections.namedtuple('struct' , 'Image CropMask ThalamusMask Header Affine InputAddress')

def loadingImage(NucleusName , dir):

    im = nib.load(dir + '/WMnMPRAGE_bias_corr.nii.gz')
    CropMsk = nib.load(dir + '/MyCrop2_Gap20.nii.gz').get_data() 

    TEMP = 1
    if TEMP == 1:
        ThalamusMsk = CropMsk.copy()

    # if '1-THALAMUS' not in NucleusName:
    #     ThalamusMsk = nib.load(dir + '/MyCrop2_Gap20.nii.gz')
    # else:
    #     ThalamusMsk = nib.load(dir + '/Thalamus.nii.gz')

    class Input:
        Image = im.get_data()
        CropMask = CropMsk
        ThalamusMask = ThalamusMsk
        Header = im.header 
        Affine = im.affine 
        InputAddress = dir

    return Input

def mainloadingImage(params):

    FullData = list()
    if  not params.MultipleTest:
        structList = loadingImage(params.NucleusName , params.directories.Input )
        FullData.append(structList)
    else:
        subFolders = listSubFolders(params.directories.Input)
        for sFi in range(5): # len(subFolders)):
            structList = loadingImage(params.NucleusName , params.directories.Input + '/' + subFolders[sFi])
            FullData.append(structList)

    return FullData
