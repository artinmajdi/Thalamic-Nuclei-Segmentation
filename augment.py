import numpy as np
import os 
from random import shuffle
from smallCodes import mkDir

def indexFunc(L,AugmentLength, ind):

    if AugmentLength > L:
        AugmentLength = L-1

    rang = range(L)
    rang = np.delete(rang,rang.index(ind))
    shuffle(rang)
    rang = rang[:AugmentLength]
    return rang

def LinearFunc(params, AugLen):

    print('---')
    # names = 'normal'
    # for AugIx in range(params.augmentLength):
    #     names = np.append( names , str(AugIx))

    # params.augmentDescription = names

def NonLinearFunc(Input, AugLen):

    Subjects = Input.Subjects
    SubjectNames = list(Subjects.keys())
    L = len(SubjectNames)

    for imInd in range(L):
        nameSubject = SubjectNames[imInd]
        subject  = Subjects[nameSubject]

        lst = indexFunc(L, AugLen, imInd)

        AugIx = 0
        for augInd in lst:
            AugIx = AugIx + 1
            print('image',imInd,'/',L,'augment',AugIx,'/',len(lst))

            nameSubjectRef = SubjectNames[augInd]
            subjectRef = Subjects[nameSubjectRef]

            Image     = subject.Address    + '/' + subject.Cropped + '.nii.gz'           
            Reference = subjectRef.Address + '/' + subjectRef.Cropped + '.nii.gz' 
            newFolder = mkDir( Input.Address + '/' + nameSubject + '_Aug' + str(AugIx) + '_Ref_' + nameSubjectRef )
            Output    = newFolder + '/' + subject.Cropped + '.nii.gz' 
            mkDir(newFolder + '/deformation')
            testwarp = newFolder + '/deformation/testWarp.nii.gz' 

            if not os.path.isfile(Output):
                os.system("ANTS 3 -m CC[%s, %s,1,5] -t SyN[0.25] -r Gauss[3,0] -o %s -i 30x90x20 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000"%(Image , Reference , newFolder + '/deformation/test') )
                os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(Image , Output , Image , testwarp) )


            Mask     = subject.Nucleus.Address    + '/' + subject.Nucleus.Cropped + '.nii.gz'           
            Reference = subjectRef.Nucleus.Address + '/' + subjectRef.Nucleus.Cropped + '.nii.gz' 
            Output    = newFolder + '/Manual_Delineation_Sanitized/' + subject.Nucleus.Cropped + '.nii.gz' 
            mkDir(newFolder + '/Manual_Delineation_Sanitized')

            if not os.path.isfile(Output):
                os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(Mask , Output , Mask , testwarp) )


def augmentMain(params , Flag):

    if params.preprocess.Augment.Mode and (params.preprocess.Augment.Rotation or params.preprocess.Augment.Shift) and (Flag == 'Linear'):
        LinearFunc(params.directories.Train.Input , params.preprocess.Augment.LinearAugmentLength)
        
    elif params.preprocess.Augment.Mode and params.preprocess.Augment.NonRigidWarp and (Flag == 'NonLinear'):
        NonLinearFunc(params.directories.Train.Input , params.preprocess.Augment.NonLinearAugmentLength)
        
        