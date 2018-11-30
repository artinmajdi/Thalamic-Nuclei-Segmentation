import numpy as np
import os 
from random import shuffle
from smallFuncs import mkDir, saveImage
from scipy.misc import imrotate
import nibabel as nib
from BashCallingFunctions import Bash_AugmentNonLinear

def funcRotating(files):

    angle = np.random.random_integers(10)

    for i in range(files.Image.shape[2]):
        files.Image[...,i] = imrotate(files.Image[...,i],angle)
        files.Mask[...,i] = imrotate(files.Mask[...,i],angle)

    return files, angle

def funcShifting(files):

    shftX = np.random.random_integers(10)
    shftY = np.random.random_integers(10)

    files.Image = np.roll(files.Image,shftX,axis=0)
    files.Image = np.roll(files.Image,shftY,axis=1)

    files.Mask = np.roll(files.Mask,shftX,axis=0)
    files.Mask = np.roll(files.Mask,shftY,axis=1)

    return files, [shftX, shftY]

def indexFunc(L,AugmentLength, ind):

    if AugmentLength > L:
        AugmentLength = L-1

    rang = range(L)
    rang = np.delete(rang,rang.index(ind))
    shuffle(rang)
    rang = rang[:AugmentLength]
    return rang

def LinearFunc(Input, Augment):

    angle = 0
    shift = [0,0]
    class files:
        Image = ''
        Mask = ''

    Subjects = Input.Subjects
    SubjectNames = list(Subjects.keys())
    L = len(SubjectNames)

    for imInd in range(L):
        nameSubject = SubjectNames[imInd]
        subject  = Subjects[nameSubject]

        # lst = indexFunc(L, AugLen, imInd)


        for AugIx in range(Augment.LinearAugmentLength):
            print('image',imInd,'/',L,'augment',AugIx,'/',Augment.LinearAugmentLength)

            im = nib.load(subject.Address + '/' + subject.ImageProcessed + '.nii.gz')  # 'Cropped' for cropped image
            files.Image  = im.get_data()
            files.Header = im.header
            files.Affine = im.affine
            files.Mask   = nib.load(subject.Label.Address + '/' + subject.Label.LabelProcessed + '.nii.gz').get_data() # 'Cropped' for cropped image

            if Augment.Rotation:
                files, angle = funcRotating(files)
            if Augment.Shift:
                files, shift = funcShifting(files)

            outDirectoryImage = mkDir( Input.Address + '/' + nameSubject + '_Aug' + str(AugIx) + '_Rot_' + str(angle) + '_shift_' + str(shift[0]) + '-' + str(shift[1]) )
            outDirectoryImage = outDirectoryImage + '/' + subject.ImageProcessed + '.nii.gz'
            outDirectoryMask = mkDir( Input.Address + '/' + nameSubject + '_Aug' + str(AugIx) + '_Rot_' + str(angle) + '_shift_' + str(shift[0]) + '-' + str(shift[1]) + '/Manual_Delineation_Sanitized')
            outDirectoryMask = outDirectoryMask + '/' + subject.Label.LabelProcessed + '.nii.gz'
            
            saveImage(files.Image , files.Affine , files.Header , outDirectoryImage)
            saveImage( np.float32(files.Mask > 0.5) , files.Affine , files.Header , outDirectoryMask)

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

            # Image     = subject.Address    + '/' + subject.ImageProcessed + '.nii.gz'           
            # Reference = subjectRef.Address + '/' + subjectRef.ImageProcessed + '.nii.gz' 

            # Mask      = subject.Label.Address    + '/' + subject.Label.LabelProcessed + '.nii.gz'           
            # Reference = subjectRef.Label.Address + '/' + subjectRef.Label.LabelProcessed + '.nii.gz' 

            outputAddress = mkDir( Input.Address + '/' + nameSubject + '_Aug' + str(AugIx) + '_Ref_' + nameSubjectRef ) 

            Bash_AugmentNonLinear(subject , subjectRef , outputAddress)

def augmentMain(params , Flag):

    if params.preprocess.Mode and params.preprocess.Augment.Mode and (params.preprocess.Augment.Rotation or params.preprocess.Augment.Shift) and (Flag == 'Linear'):
        LinearFunc(params.directories.Train.Input , params.preprocess.Augment)
        
    elif params.preprocess.Mode and params.preprocess.Augment.Mode and params.preprocess.Augment.NonRigidWarp and (Flag == 'NonLinear'):
        NonLinearFunc(params.directories.Train.Input , params.preprocess.Augment)
        
        