import os
import numpy as np

def RigidRegistration(subject , Template):

    BiasCorrected = subject.Address + '/' + subject.origImage + '_bias_corr.nii.gz'
    outP = subject.Address + '/CropMask.nii.gz'

    if ( not os.path.isfile( outP ) )  and ( '_bias_corr_Cropped' not in subject.Cropped ):
        os.system("ANTS 3 -m CC[%s, %s ,1,5] -o linear -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(BiasCorrected , Template.Image) )
        os.system("WarpImageMultiTransform 3 %s %s -R %s linearAffine.txt"%(Template.Mask , outP , BiasCorrected) )

def BiasCorrection(subject):

    inP  = subject.Address + '/' + subject.origImage + '.nii.gz'
    outP = subject.Address + '/' + subject.origImage + '_bias_corr.nii.gz'
    
    if ( not os.path.isfile(outP) ) and ( '_bias_corr_Cropped' not in subject.Cropped ):
        os.system("N4BiasFieldCorrection -d 3 -i %s -o %s -b [200] -s 3 -c [50x50x30x20,1e-6]"%( inP, outP )  )

def Cropping(subject):

    inP  = subject.Address + '/' + subject.origImage + '_bias_corr.nii.gz'
    outP = subject.Address + '/' + subject.origImage + '_bias_corr_Cropped.nii.gz'
    crop = subject.Address + '/' + 'CropMask.nii.gz'

    if ( not os.path.isfile( outP ) ) and ( '_bias_corr_Cropped' not in subject.Cropped ):
        os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

    # Cropping the Label
    inP  = subject.Nucleus.Address + '/' + subject.Nucleus.Full + '.nii.gz'
    outP = subject.Nucleus.Address + '/' + subject.Nucleus.Full + '_Cropped.nii.gz'
    crop = subject.Address + '/' + 'CropMask.nii.gz'

    if ( not os.path.isfile( outP ) ) and ( '_bias_corr_Cropped' not in subject.Cropped ):
        os.system("ExtractRegionFromImageByMask 3 %s %s %s 1 0"%( inP , outP , crop ) )

# def NonLinearFunc(Subjects,AugLen):

#     SubjectNames = list(Subjects.keys())
#     L = len(SubjectNames)

#     for imInd in range(L):
#         nameSubject = SubjectNames[imInd]
#         subject  = Subjects[nameSubject]

#         lst = indexFunc(L,AugLen, imInd)

#         for augInd in lst:
#             nameSubjectRef = SubjectNames[augInd]
#             subjectRef = Subjects[nameSubjectRef]

#             Image     = subject.Address    + '/' + subject.origImage    + '_bias_corr_Cropped.nii.gz'           
#             Reference = subjectRef.Address + '/' + subjectRef.origImage + '_bias_corr_Cropped.nii.gz'
#             Output    = subject.Address    + '/' + subject.origImage    + '_bias_corr_Cropped_Aug' + str(augInd) + '.nii.gz'

        
#         #     Image = 'crop_LIFUP004_MPRAGE_WMn.nii.gz'
#         #     Reference = '../case3/crop_LIFUP003_MPRAGE_WMn.nii.gz'
#         #     Output = 'aug1.nii.gz'
#             os.system("ANTS 3 -m CC[%s, %s,1,5] -t SyN[0.25] -r Gauss[3,0] -o test -i 30x90x20 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000"%(Image , Reference) )
#             os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t testWarp.nii.gz"%(Image , Output , Image) )
