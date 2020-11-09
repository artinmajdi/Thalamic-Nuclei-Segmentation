import os
import sys
import pathlib
from nilearn import image as niImage
import nibabel as nib
import json
from shutil import copyfile
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from otherFuncs import smallFuncs
from Parameters import paramFunc, UserInfo

def main(params):
    def loop_subjects(Mode):

        class Info:
            mode = Mode
            dirr = params.directories.Train if Mode == 'train' else params.directories.Test
            Length = len(dirr.Input.Subjects)
            subjectName = ''
            ind = ''
            Subjects = dirr.Input.Subjects

        for Info.ind, Info.subjectName in enumerate(Info.Subjects):
            apply_On_Individual(params, Info)

    if params.preprocess.Mode:
        if not params.WhichExperiment.TestOnly._mode:
            loop_subjects('train')

        loop_subjects('test')


def apply_On_Individual(params, Info):
    print('(' + str(Info.ind) + '/' + str(Info.Length) + ')', Info.mode, Info.subjectName)
    # This if statement skips the augmented data, assuming that, the augmented data was made frmo the already preprocessed input
    if 'Aug' not in Info.subjectName:
        subject = Info.Subjects[Info.subjectName]

        # Duplicating the original nifti image with name "*_PProcessed.nii.gz"
        duplicating_original_files_as_PProcessed(subject, params)

        if params.preprocess.Cropping:
            print('     Rigid Registration')
            RigidRegistration(subject, params.WhichExperiment.HardParams.Template)

            print('     Cropping')
            func_cropImage(params, subject)

        if params.preprocess.BiasCorrection:
            print('     Bias Correction')
            BiasCorrection(subject, params)

        if params.preprocess.Reslicing:
            print('     ReSlicing')
            apply_reslice(subject, params)

    return params


def apply_reslice(subject, params):
    class Reference:
        def __init__(self, nucleus='Image'):

            self.dir_origRefImage = 'path-to-reference-image'
            self.dir = params.WhichExperiment.Experiment._code_address + '/general/Reslicing/'
            self.nucleus = nucleus if not ('.nii.gz' in nucleus) else nucleus.split('.nii.gz')[0]

        def write(self):

            if self.nucleus == 'Image':
                DirT = 'WMnMPRAGE_bias_corr.nii.gz'
            else:
                DirT = 'Label/' + self.nucleus + '.nii.gz'

            if os.path.exists(self.dir_origRefImage + DirT):
                ref = nib.load(self.dir_origRefImage + DirT)

                Info_Ref = {'affine': ref.affine.tolist(), 'shape': ref.shape}
                with open(self.dir + self.nucleus + '.json', "w") as j:
                    j.write(json.dumps(Info_Ref))
            else:
                print('nucleus %s doesn not exist' % self.nucleus)

        def read(self):
            if os.path.exists(self.dir + self.nucleus + '.json'):

                with open(self.dir + self.nucleus + '.json', "r") as j:
                    info = json.load(j)

                    info['affine'] = np.array(info['affine'])
                    info['shape'] = tuple(info['shape'])

                    return info
            else:
                print('nucleus %s doesn not exist' % self.nucleus)

        def write_all_nuclei(self):
            for self.nucleus in np.append('Image', params.WhichExperiment.Nucleus.FullNames):
                Reference(self.nucleus).write()

    # Reading the reference transformations that the target nifti file will be warped into
    target_affine = Reference(nucleus='Image').read()['affine'][:3, :3]

    def apply_reslicing_main(input_image, output_image, outDebug, interpolation):
        """ Applyingthe re-slicing or mapping of the input image into the reference resolution 

        Args:
            input_image   (str): Path to the input image
            output_image  (str): Path to the output image
            outDebug      (str): Path to the debug file (re-slicied image)
            interpolation (str): Mode of interpolation
            ref     (Reference): Reference Affine matrix that includes the resolution & amount of shift
        """

        # Checking if the input file exist. Mostly relevant for preprocessing cases without manual labels
        if not os.path.isfile(input_image):
            pass

        # Checks to see if an already re-sliced file exists inside the debug folder (temp)      
        elif os.path.isfile(outDebug):
            copyfile(outDebug, output_image)

        # If there wasn't an existng re-sliced nifti file, it will apply the re-slicing onto the input image
        else:
            # Re-sampling the input image
            im = niImage.resample_img(img=nib.load(input_image), target_affine=target_affine, interpolation=interpolation)

            # Saving the resampled image
            nib.save(im, output_image)

            # Copying the resampled image into the debug folder (temp)
            copyfile(output_image, outDebug)

    def apply_to_Image(subject):

        # Path to the input nifti file
        inP  = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
        outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
        outDebug = smallFuncs.mkDir(subject.Temp.address + '/') + subject.ImageOriginal + '_resliced.nii.gz'

        # Re-scliing the input nifti image
        apply_reslicing_main(inP, outP, outDebug, 'continuous')

    def apply_to_mask(subject):

        for nucleus_name in params.WhichExperiment.Nucleus.FullNames:
            
            # Path to the input nifti file
            inP  = subject.Label.address + '/' + nucleus_name + '_PProcessed.nii.gz'
            outP = subject.Label.address + '/' + nucleus_name + '_PProcessed.nii.gz'
            outDebug = subject.Label.Temp.address + '/' + nucleus_name + '_resliced.nii.gz'

            # Mapping the input nifti file into the references resolution.
            apply_reslicing_main(inP, outP, outDebug, 'nearest')

    # Applying the re-slicing on the input image
    apply_to_Image(subject)

    # Applying the re-slicing on thalamic nuclei
    apply_to_mask(subject)


def RigidRegistration(subject, Template):
    processed = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.Temp.address + '/CropMask.nii.gz'
    LinearAffine = subject.Temp.Deformation.address + '/linearAffine.txt'

    if not os.path.isfile(LinearAffine):
        os.system(
            "ANTS 3 -m CC[%s, %s ,1,5] -o %s -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" % (
                processed, Template.Image, subject.Temp.Deformation.address + '/linear'))

    if not os.path.isfile(outP):
        os.system("WarpImageMultiTransform 3 %s %s -R %s %s" % (Template.Mask, outP, processed, LinearAffine))


def BiasCorrection(subject, params):
    inP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
    outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_bias_corr.nii.gz'

    if os.path.isfile(outDebug):
        copyfile(outDebug, outP)
    else:
        os.system("N4BiasFieldCorrection -d 3 -i %s -o %s -b [200] -s 3 -c [50x50x30x20,1e-6]" % (inP, outP))
        if params.preprocess.save_debug_files:
            copyfile(outP, outDebug)


def func_cropImage(params, subject):
    def finding_boundingbox_encompassing_crop_mask(CropMask, Gap):
        """ Finding the coordinates for the boundingbox encompassing the cropped mask

        Args:
            CropMask (numpy array): input cropping mask
            Gap              (int): This number specifies whether and how much the final boundingbox should exceed the original cropping mask

        Returns:
            d        (numpy array): Boundingbox coordinates
        """
        # Finding the boundingbox that encompassed the TRUE area of the cropped mask
        BBCord = smallFuncs.findBoundingBox(CropMask > 0.5)

        # Converting the boundingbox coordinates into appropriate coordinate vector
        d = np.zeros((3, 2), dtype=np.int)
        for ix in range(len(BBCord)):
            d[ix, :] = [BBCord[ix][0] - Gap[ix], BBCord[ix][-1] + Gap[ix]]
            d[ix, :] = [max(d[ix, 0], 0), min(d[ix, 1], CropMask.shape[ix])]

        return d

    # The cropping mask & it's boundingbox coordinates
    crop = subject.Temp.address + '/CropMask.nii.gz'
    crop_cord = finding_boundingbox_encompassing_crop_mask(nib.load(crop).get_fdata(), [0, 0, 0])

    def check_crop(inP, outP, outDebug):
        """ Cropping the input using the cropped mask

        Args:
            inP                     (str): Address to the input image
            outP                    (str): Address to the output image
            outDebug                (str): Address to the cropped nifti image inside the temp folder
        """

        # If an already cropped nifti image exists inside the temp folder, this will copy that into the 
        # main image directory and replace the original image
        if os.path.isfile(outDebug):
            copyfile(outDebug, outP)

        # If there isn't an already cropped image inside the debug subfolder (temp folder), this will crop 
        # the input image using the boundingbox coordinates
        elif os.path.isfile(inP):

            # Cropping the input image
            mskC = nib.load(inP).slicer[crop_cord[0, 0]:crop_cord[0, 1], 
                                        crop_cord[1, 0]:crop_cord[1, 1], 
                                        crop_cord[2, 0]:crop_cord[2, 1]]

            # Saving the input image
            nib.save(mskC, outP)

            # Saving the newly cropped image into the debug subfolder (temp folder)
            if params.preprocess.save_debug_files:
                copyfile(outP, outDebug)
        else:
            raise Exception('*_PProcessed.nii.gz does not exist. This should have been created automatically')

    # Setting the input & output image address and the path to cropped image inside the debug folder
    inP = outP = subject.address    + '/' + subject.ImageProcessed + '.nii.gz'
    outDebug = subject.Temp.address + '/' + subject.ImageOriginal  + '_Cropped.nii.gz'

    # Cropping the input image using the boundingbox coordinates
    check_crop(inP, outP, outDebug)

    # Looping through all thalamic nuclei
    for nucleus_name in params.WhichExperiment.Nucleus.FullNames:

        # Finding the directory to each nucleus
        inP = outP = subject.Label.address    + '/' + nucleus_name + '_PProcessed.nii.gz'
        outDebug = subject.Label.Temp.address + '/' + nucleus_name + '_Cropped.nii.gz'

        # Cropping the nucleus mask using the broundingbox coordinates
        check_crop(inP, outP, outDebug)


def duplicating_original_files_as_PProcessed(subject, params):
    copyfile(subject.address + '/' + subject.ImageOriginal + '.nii.gz', subject.address + '/PProcessed.nii.gz')

    for nucleus_name in params.WhichExperiment.Nucleus.FullNames:
        copyfile(subject.Label.address + '/' + nucleus_name + '.nii.gz', subject.Label.address + '/' + nucleus_name + '_PProcessed.nii.gz')



if __name__ == "__main__":
    UserEntry = smallFuncs.terminalEntries(UserInfo.__dict__)
    UserEntry['preprocess'].Mode = True
    user_parameters = paramFunc.Run(UserEntry)
    main(user_parameters)