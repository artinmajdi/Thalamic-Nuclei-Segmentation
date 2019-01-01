import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from random import shuffle
from otherFuncs.smallFuncs import mkDir, saveImage , NucleiSelection , copyfile
from scipy.misc import imrotate
import nibabel as nib
from preprocess.BashCallingFunctions import Bash_AugmentNonLinear

def funcRotating(Image , angle):

    for i in range(Image.shape[2]):
        Image[...,i] = imrotate(Image[...,i],angle)

    return Image

def funcShifting(Image , shift):

    Image = np.roll(Image,shift[0],axis=0)
    Image = np.roll(Image,shift[1],axis=1)

    return Image

def indexFunc(L,AugmentLength, ind):

    if AugmentLength > L:
        AugmentLength = L-1

    rang = range(L)
    rang = np.delete(rang,rang.index(ind))
    shuffle(rang)
    rang = rang[:AugmentLength]
    return rang

def LinearFunc(params):

    Subjects = params.directories.Train.Input.Subjects
    SubjectNames = list(Subjects.keys())
    L = len(SubjectNames)

    for imInd in range(L):
        nameSubject = SubjectNames[imInd]
        subject  = Subjects[nameSubject]

        # lst = indexFunc(L, AugLen, imInd)
        for AugIx in range(params.preprocess.Augment.LinearAugmentLength):
            print('image',imInd,'/',L,'augment',AugIx,'/',params.preprocess.Augment.LinearAugmentLength)

            im = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')  # 'Cropped' for cropped image
            Image  = im.get_data()
            Header = im.header
            Affine = im.affine

            angle = np.random.random_integers(10)
            shift = [ np.random.random_integers(10) , np.random.random_integers(10)]

            outDirectoryImage = mkDir( params.directories.Train.Input.address + '/' + nameSubject + '_Aug' + str(AugIx) + '_Rot_' + str(angle) + '_shift_' + str(shift[0]) + '-' + str(shift[1]) )
            outDirectoryMask = mkDir( params.directories.Train.Input.address + '/' + nameSubject + '_Aug' + str(AugIx) + '_Rot_' + str(angle) + '_shift_' + str(shift[0]) + '-' + str(shift[1]) + '/Labels')

            if params.preprocess.Augment.Rotation:
                Image = funcRotating(Image , angle)
            if params.preprocess.Augment.Shift:
                Image = funcShifting(Image , shift)

            outDirectoryImage2 = outDirectoryImage + '/' + subject.ImageProcessed + '.nii.gz'
            saveImage(Image , Affine , Header , outDirectoryImage2)
            copyfile(outDirectoryImage2 , outDirectoryImage  + '/' + subject.ImageProcessed.split('_PProcessed')[0] + '.nii.gz')

            for ind in params.directories.WhichExperiment.Nucleus.FullIndexes:
                NucleusName, _ = NucleiSelection(ind , params.directories.WhichExperiment.Nucleus.Organ)

                Mask   = nib.load(subject.Label.address + '/' + NucleusName + '_PProcessed.nii.gz').get_data() # 'Cropped' for cropped image

                if params.preprocess.Augment.Rotation:
                    Mask = funcRotating(Mask , angle)
                if params.preprocess.Augment.Shift:
                    Mask = funcShifting(Mask , shift)

                outDirectoryMask2  = outDirectoryMask  + '/' + NucleusName + '_PProcessed.nii.gz'
                saveImage( np.float32(Mask > 0.5) , Affine , Header , outDirectoryMask2)
                copyfile(outDirectoryMask2 , outDirectoryMask  + '/' + NucleusName + '.nii.gz')


def NonLinearFunc(Input, Augment):

    Subjects = Input.Subjects
    SubjectNames = list(Subjects.keys())
    L = len(SubjectNames)

    for imInd in range(L):
        nameSubject = SubjectNames[imInd]
        subject  = Subjects[nameSubject]

        lst = indexFunc(L, Augment.NonLinearAugmentLength, imInd)

        AugIx = 0
        for augInd in lst:
            AugIx = AugIx + 1
            print('image',imInd,'/',L,'augment',AugIx,'/',len(lst))

            nameSubjectRef = SubjectNames[augInd]
            subjectRef = Subjects[nameSubjectRef]

            # Image     = subject.address    + '/' + subject.ImageProcessed + '.nii.gz'
            # Reference = subjectRef.address + '/' + subjectRef.ImageProcessed + '.nii.gz'

            # Mask      = subject.Label.address    + '/' + subject.Label.LabelProcessed + '.nii.gz'
            # Reference = subjectRef.Label.address + '/' + subjectRef.Label.LabelProcessed + '.nii.gz'

            outputAddress = mkDir( Input.address + '/' + nameSubject + '_Aug' + str(AugIx) + '_Ref_' + nameSubjectRef )

            Bash_AugmentNonLinear(subject , subjectRef , outputAddress)

def augmentMain(params , Flag):

    if params.preprocess.Mode and params.preprocess.Augment.Mode and (params.preprocess.Augment.Rotation or params.preprocess.Augment.Shift) and (Flag == 'Linear'):
        LinearFunc(params)

    elif params.preprocess.Mode and params.preprocess.Augment.Mode and params.preprocess.Augment.NonRigidWarp and (Flag == 'NonLinear'):
        NonLinearFunc(params.directories.Train.Input , params.preprocess.Augment)
