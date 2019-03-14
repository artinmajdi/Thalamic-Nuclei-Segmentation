# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from random import shuffle
from scipy.misc import imrotate
import nibabel as nib
import preprocess.BashCallingFunctionsA as BashCallingFunctionsA
import otherFuncs.smallFuncs as smallFuncs
import os
import skimage
import numpy as np


def funcShearing(Image, Shear, Order):    
    inverse_map = skimage.transform.AffineTransform(shear= np.deg2rad(Shear)) #  * 0.01745   # 0.01745 = pi/180

    if np.random.randint(low=0, high=2) == 1: inverse_map = inverse_map.inverse
    for i in range(Image.shape[2]):        
        Image[...,i] = skimage.transform.warp(Image[...,i], inverse_map=inverse_map, order=Order, clip=True, preserve_range=True,output_shape=tuple(Image.shape[:2]))
        
    return Image   

def funcScaling(Image, Scale):    
    for i in range(Image.shape[2]):
        Image[...,i] = skimage.transform.AffineTransform(Image[...,i] , scale=Scale) 
    return Image 

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

def LinearFunc(params, mode):

    def nameOutput(nameSubject, InputThreshs, AugIx):
        nameSubject2 = nameSubject + '_Aug' + str(AugIx)
        if params.Augment.Linear.Rotation.Mode: nameSubject2 += '_Rot_'   + str(InputThreshs.angle) 
        if params.Augment.Linear.Shift.Mode:    nameSubject2 += '_shift_' + str(InputThreshs.shift[0]) + '-' + str(InputThreshs.shift[1])
        if params.Augment.Linear.Shear.Mode:    nameSubject2 += '_Shear_' + str(InputThreshs.Shear) 
        return nameSubject2
        
    def inputThresholds():
        angleMax = params.Augment.Linear.Rotation.AngleMax
        shiftMax = params.Augment.Linear.Shift.ShiftMax              
        ShearMax = params.Augment.Linear.Shear.ShearMax

        class InputThreshs:
            angle = np.random.random_integers(-angleMax,angleMax)
            shift = [ np.random.random_integers(-shiftMax,shiftMax) , np.random.random_integers(-shiftMax,shiftMax)]
            Shear = np.random.random_integers(-ShearMax,ShearMax)

        return InputThreshs
            
    def applyLinearAugment(im, InputThreshs, mode):
        if params.Augment.Linear.Rotation.Mode: im = funcRotating(im , InputThreshs.angle)
        if params.Augment.Linear.Shift.Mode:    im = funcShifting(im , InputThreshs.shift)
        if params.Augment.Linear.Shear.Mode:    im = funcShearing(im , InputThreshs.Shear, mode)
        return im
                
    Subjects = params.directories.Train.Input.Subjects
    SubjectNames = list(Subjects.keys())
    L = len(SubjectNames)

    for imInd in range(L):
        nameSubject = SubjectNames[imInd]
        subject  = Subjects[nameSubject]

        for AugIx in range(params.Augment.Linear.Length):
            print('image',imInd,'/',L,'augment',AugIx,'/',params.Augment.Linear.Length)

            imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')  # 'Cropped' for cropped image

            InputThreshs = inputThresholds()
            nameSubject2 = nameOutput(nameSubject, InputThreshs, AugIx)

            Image = applyLinearAugment(imF.get_data(), InputThreshs, 3)

            class outDirectory:
                Image = smallFuncs.mkDir( params.directories.Train.Input.address + '/' + nameSubject2)
                Mask = smallFuncs.mkDir( params.directories.Train.Input.address + '/' + nameSubject2 + '/Labels')

            smallFuncs.saveImage(Image , imF.affine , imF.header , outDirectory.Image + '/' + subject.ImageProcessed + '.nii.gz' )
            smallFuncs.mkDir(outDirectory.Image + '/temp/')

            subF = [s for s in os.listdir(subject.Label.address) if 'PProcessed' in s]
            for NucleusName in subF: 

                Mask = nib.load(subject.Label.address + '/' + NucleusName).get_data() 
                Mask = applyLinearAugment(Mask, InputThreshs, 1)

                smallFuncs.saveImage( np.float32(Mask > 0.5) , imF.affine , imF.header ,  outDirectory.Mask  + '/' + NucleusName  )

def NonLinearFunc(Input, Augment, mode):

    Subjects = Input.Subjects
    SubjectNames = list(Subjects.keys())
    L = len(SubjectNames)

    for imInd in range(L):
        nameSubject = SubjectNames[imInd]
        subject  = Subjects[nameSubject]

        lst = indexFunc(L, Augment.NonLinear.Length, imInd)

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

            outputAddress = smallFuncs.mkDir( Input.address + '/' + nameSubject + '_Aug' + str(AugIx) + '_Ref_' + nameSubjectRef )

            BashCallingFunctionsA.Bash_AugmentNonLinear(subject , subjectRef , outputAddress)

# TODO fix "LinearFunc" & "NonLinearFunc" function to count for situations when we only want to apply the function on one case
def main_augment(params , Flag, mode):

    if params.Augment.Mode and (params.Augment.Linear.Rotation.Mode or params.Augment.Linear.Shift.Mode or params.Augment.Linear.Shear.Mode)  and (Flag == 'Linear'):
        LinearFunc(params, mode)

    elif params.Augment.Mode and params.Augment.NonLinear.Mode and (Flag == 'NonLinear'):
        NonLinearFunc(params.directories.Train.Input , params.Augment, mode)

