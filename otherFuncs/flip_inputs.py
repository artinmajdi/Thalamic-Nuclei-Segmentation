import nibabel as nib
import sys
import os


def terminalEntries():
    for en in range(len(sys.argv)):
        entry = sys.argv[en]

        if entry.lower() in ('-i', '--input'):  # gpu num
            dir_input = sys.argv[en + 1]

        elif entry in ('-o', '--output'):
            dir_output = sys.argv[en + 1]

    return dir_input, dir_output

def mkDir(Dir):
    if not os.path.isdir(Dir):
        os.makedirs(Dir)
    return Dir

def saveImage(Image, Affine, Header, outDirectory):
    """ Inputs:  Image , Affine , Header , outDirectory """
    mkDir(outDirectory.split(os.path.basename(outDirectory))[0])
    out = nib.Nifti1Image(Image.astype('float32'), Affine)
    out.get_header = Header
    nib.save(out, outDirectory)

if __name__ == "__main__":
    dir_input, dir_output = terminalEntries()
    orientation, nucleus = dir_input.split('/')[-2:]
    if orientation == 'sd0':        orientation = 'Sagittal'
    elif  orientation == 'sd1':     orientation = 'Coronal'
    elif  orientation == 'sd2':     orientation = 'Axial'
    elif  orientation == '2.5D_MV': orientation = 'Majority Voting'

    try:
        nucleus = nucleus.split('.nii.gz')[0].split('-',1)[1]
    except:
        ''


    print(f"    Flipping: {orientation}  {nucleus}")
    imageF = nib.load(dir_input)
    image = imageF.get_fdata()[::-1,:,:]
    saveImage(image, imageF.affine, imageF.header, dir_output)
