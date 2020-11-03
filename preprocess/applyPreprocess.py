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
        # params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
        if not params.WhichExperiment.TestOnly.mode:
            loop_subjects('train')

        loop_subjects('test')


def apply_On_Individual(params, Info):
    print('(' + str(Info.ind) + '/' + str(Info.Length) + ')', Info.mode, Info.subjectName)
    # This if statement skips the augmented data, assuming that, the augmented data was made frmo the already preprocessed input
    if 'Aug' not in Info.subjectName:
        subject = Info.Subjects[Info.subjectName]

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

            self.dir_origRefImage = '/array/ssd/msmajdi/experiments/keras/exp3/train/Main/vimp2_819_05172013_DS/'
            self.dir = params.WhichExperiment.Experiment.code_address + '/general/Reslicing/'
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
            for self.nucleus in np.append('Image', smallFuncs.Nuclei_Class().All_Nuclei().Names):
                Reference(self.nucleus).write()

    def apply_reslicing_main(input_image, output_image, outDebug, interpolation, ref):
        """ Applyingthe re-slicing or mapping of the input image into the reference resolution 

        Args:
            input_image   (str): Path to the input image
            output_image  (str): Path to the output image
            outDebug      (str): Path to the debug file (re-slicied image)
            interpolation (str): Mode of interpolation
            ref     (Reference): Reference Affine matrix that includes the resolution & amount of shift
        """

        # Checks to see if an already re-sliced file exists inside the debug folder (temp)      
        if os.path.isfile(outDebug):
            copyfile(outDebug, output_image)

        # If there wasn't an existng re-sliced nifti file, it will apply the re-slicing onto the input image
        else:

            # Re-sampling the input image
            im = niImage.resample_img(img=nib.load(input_image), target_affine=ref['affine'][:3, :3],
                                      interpolation=interpolation)

            # Saving the resampled image
            nib.save(im, output_image)

            # Copying the resampled image into the debug folder (temp)
            copyfile(output_image, outDebug)

    def apply_to_Image(subject):
        # Reading the reference transformations that the target nifti file will be warped into
        ref = Reference(nucleus='Image').read()

        # Path to the input nifti file
        input_image = subject.address + '/' + subject.ImageProcessed + '.nii.gz'

        # Path to the output nifti file
        output_image = subject.address + '/' + subject.ImageProcessed + '.nii.gz'

        # Path to the debug nifti file that will be or already is saved inside the temp subfolder
        outDebug = smallFuncs.mkDir(subject.Temp.address + '/') + subject.ImageOriginal + '_resliced.nii.gz'

        # Re-scliing the input nifti image
        apply_reslicing_main(input_image, output_image, outDebug, 'continuous', ref)

    def apply_to_mask(subject):
        ref = Reference(nucleus=nucleus).read()

        if subject.Label.address:
            input_nucleus = subject.Label.address + '/' + nucleus + '_PProcessed.nii.gz'
            output_nucleus = subject.Label.address + '/' + nucleus + '_PProcessed.nii.gz'
            outDebug = subject.Label.Temp.address + '/' + nucleus + '_resliced.nii.gz'

            # Checking if the nucleus nifti file exist inside the subject folder
            if os.path.isfile(input_nucleus):
                # Mapping the input nifti file into the references resolution.
                apply_reslicing_main(input_nucleus, output_nucleus, outDebug, 'nearest', ref)

    # Applying the re-slicing on the input image
    apply_to_Image(subject)

    # Applying the re-slicing on thalamic nuclei
    for nucleus in smallFuncs.Nuclei_Class().Names:
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
    def cropImage_FromCoordinates(CropMask, Gap):
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

    crop = subject.Temp.address + '/CropMask.nii.gz'

    def check_crop(inP, outP, outDebug, CropCoordinates):
        """ Cropping the input using the cropped mask

        Args:
            inP                     (str): Address to the input image
            outP                    (str): Address to the output image
            outDebug                (str): Address to the cropped nifti image inside the temp folder
            CropCoordinates (numpy array): Boundingbox coordinates encompassing the cropped mask
        """

        def applyCropping(image):
            d = CropCoordinates
            return image.slicer[d[0, 0]:d[0, 1], d[1, 0]:d[1, 1], d[2, 0]:d[2, 1]]

        # If an already cropped nifti image exists inside the temp folder, this will copy that into the 
        # main image directory and replace the original image
        if os.path.isfile(outDebug):
            copyfile(outDebug, outP)

        # If there isn't an already cropped image inside the debug subfolder (temp folder), this will crop 
        # the input image using the boundingbox coordinates
        elif os.path.isfile(inP):

            # Cropping the input image
            mskC = applyCropping(nib.load(inP))

            # Saving the input image
            nib.save(mskC, outP)

            # Saving the newly cropped image into the debug subfolder (temp folder)
            if params.preprocess.save_debug_files:
                copyfile(outP, outDebug)

    def directoriesImage(subject):
        inP = outP = subject.address + '/' + subject.ImageProcessed + '.nii.gz'
        outDebug = subject.Temp.address + '/' + subject.ImageOriginal + '_Cropped.nii.gz'
        return inP, outP, outDebug

    def directoriesNuclei(subject, ind):
        NucleusName, _, _ = smallFuncs.NucleiSelection(ind)
        inP = outP = subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz'
        outDebug = subject.Label.Temp.address + '/' + NucleusName + '_Cropped.nii.gz'
        return inP, outP, outDebug

    # Setting the input & output image address and the path to cropped image inside the debug folder
    inP, outP, outDebug = directoriesImage(subject)

    # If an already cropped nifti image doesn't exist inside the temp folder this will estimate the 
    # boundingbox coordinates frmo the cropped mask
    CropCoordinates = np.array([])
    if not os.path.isfile(outDebug):
        CropCoordinates = cropImage_FromCoordinates(nib.load(crop).get_fdata(), [0, 0, 0])

    # Cropping the input image using the boundingbox coordinates
    check_crop(inP, outP, outDebug, CropCoordinates)

    # Looping through all thalamic nuclei
    for ind in params.WhichExperiment.Nucleus.FullIndexes:

        # Finding the directory to each nucleus
        inP, outP, outDebug = directoriesNuclei(subject, ind)

        # If an already cropped nucleus mask & the cropped mask coordinates doesn't exist, this will 
        # estimate the boundingbox coordinates. This applies for rare cases where there was a cropped 
        # input image, but not cropped nucleus mask
        if not os.path.isfile(outDebug) and CropCoordinates.any():
            CropCoordinates = cropImage_FromCoordinates(nib.load(crop).get_fdata(), [0, 0, 0])

        # Cropping the nucleus mask using the broundingbox coordinates
        check_crop(inP, outP, outDebug, CropCoordinates)


if __name__ == "__main__":
    user_parameters = paramFunc.Run(UserInfo.__dict__, terminal=True)
    main(user_parameters)
