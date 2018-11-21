import nibabel as nib
import collections
from smallCodes import listSubFolders

struct = collections.namedtuple('struct' , 'Image CropMask ThalamusMask Header Affine InputAddress')

def loadingImage(Dir, Files):
    
    im = nib.load(Dir + '/' + Files.Image + '.nii.gz')
    CropMsk = nib.load(Dir + '/' + Files.Image + '.nii.gz').get_data() 

    TEMP = 1
    if TEMP == 1:
        ThalamusMsk = CropMsk.copy()

    # if '1-THALAMUS' not in NucleusName:
    #     ThalamusMsk = nib.load(Dir + '/MyCrop2_Gap20.nii.gz')
    # else:
    #     ThalamusMsk = nib.load(Dir + '/Thalamus.nii.gz')

    class Input:
        Image = im.get_data()
        CropMask = CropMsk
        ThalamusMask = ThalamusMsk
        Header = im.header 
        Affine = im.affine 
        InputAddress = Dir

    return Input

def mainloadingImage(Dir, Files):

    structList = loadingImage( Dir, Files)

    # FullData = list()
    # if  not params.directories.Input.MultipleTest:
    #     structList = loadingImage(params.TrainParams.NucleusName , params.directories.Input )
    #     FullData.append(structList)
    # else:
    #     subFolders = listSubFolders(params.directories.Input.Address)
    #     for sFi in range(len(subFolders)):
    #         structList = loadingImage(params.TrainParams.NucleusName , params.directories.Input + '/' + subFolders[sFi])
    #         FullData.append(structList)

    return structList
