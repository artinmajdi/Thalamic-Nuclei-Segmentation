from random import shuffle
import nibabel as nib
import otherFuncs.smallFuncs as smallFuncs
import os
import skimage
import numpy as np

def indexFunc(L,AugmentLength, ind):

    if AugmentLength > L:
        AugmentLength = L-1

    rang = range(L)
    rang = np.delete(rang,rang.index(ind))
    shuffle(rang)
    rang = rang[:AugmentLength]
    return rang

def LinearFunc(params, mode):

    class linearFuncs_Class():    
        def __init__(self, params, nameSubject, AugIx , subject):
            self.params = params
            self.subject = subject

            def inputThresholds(self):
                angleMax = self.params.Augment.Linear.Rotation.AngleMax
                shiftMax = self.params.Augment.Linear.Shift.ShiftMax              
                ShearMax = self.params.Augment.Linear.Shear.ShearMax

                class InputThreshs:
                    angle = np.random.random_integers(-angleMax,angleMax)
                    shift = [ np.random.random_integers(-shiftMax,shiftMax) , np.random.random_integers(-shiftMax,shiftMax)]
                    Shear = np.random.random_integers(-ShearMax,ShearMax)

                if InputThreshs.angle == 0: InputThreshs.angle = np.random.random_integers(-angleMax,angleMax)
                if InputThreshs.Shear == 0: InputThreshs.Shear = np.random.random_integers(-ShearMax,ShearMax)

                self.InputThreshs = InputThreshs()
            inputThresholds(self)

            def nameOutput(self, nameSubject, AugIx):
                nameSubject2 = nameSubject + '_Aug' + str(AugIx)
                if self.params.Augment.Linear.Rotation.Mode: nameSubject2 += '_Rot_'   + str(self.InputThreshs.angle) 
                if self.params.Augment.Linear.Shift.Mode:    nameSubject2 += '_shift_' + str(self.InputThreshs.shift[0]) + '-' + str(self.InputThreshs.shift[1])
                if self.params.Augment.Linear.Shear.Mode:    nameSubject2 += '_Shear_' + str(self.InputThreshs.Shear) 
                self.nameSubject2 = nameSubject2
            nameOutput(self, nameSubject, AugIx)
                
            sd = smallFuncs.mkDir('sd' + str(params.WhichExperiment.Dataset.slicingInfo.slicingDim))
            class outDirectory:
                Image = smallFuncs.mkDir( self.params.directories.Train.Input.address + '/' + sd + '/' + self.nameSubject2 + '_' + sd)
                Mask = smallFuncs.mkDir( self.params.directories.Train.Input.address + '/' + sd + '/' + self.nameSubject2 + '_' + sd + '/Label')
                Temp = smallFuncs.mkDir( self.params.directories.Train.Input.address + '/' + sd + '/' + self.nameSubject2 + '_' + sd + '/temp')
            self.outDirectory = outDirectory()

            smallFuncs.mkDir(self.outDirectory.Image + '/temp/')

        def main(self, Image, order):
            # apply_linear = linearFuncs_Class(im).
            self.Image = Image
            self.order = order
            
            def funcShearing(self):
                inverse_map = skimage.transform.AffineTransform(shear=np.deg2rad(self.InputThreshs.Shear)) #  * 0.01745   # 0.01745 = pi/180

                if np.random.randint(low=0, high=2) == 1: inverse_map = inverse_map.inverse
                for i in range(self.Image.shape[2]):        
                    self.Image[...,i] = skimage.transform.warp(self.Image[...,i], inverse_map=inverse_map, order=self.order, clip=True, preserve_range=True,output_shape=tuple(self.Image.shape[:2]))
                
            def funcScaling(self):    
                for i in range(self.Image.shape[2]):
                    self.Image[...,i] = skimage.transform.AffineTransform(self.Image[...,i] , scale=self.InputThreshs.Scale) 

            def funcRotating(self):
                for i in range(self.Image.shape[2]): 
                    self.Image[...,i] = skimage.transform.rotate(self.Image[...,i] ,self.InputThreshs.angle, order=self.order, preserve_range=True)
                                    
            def funcShifting(self):
                self.Image = np.roll(self.Image,self.InputThreshs.shift[0],axis=0)
                self.Image = np.roll(self.Image,self.InputThreshs.shift[1],axis=1)

            # for slicingDim in params.WhichExperiment.Dataset.slicingInfo.slicingDim:
            self.Image = np.transpose(self.Image, self.params.WhichExperiment.Dataset.slicingInfo.slicingOrder)

            if self.params.Augment.Linear.Rotation.Mode: funcRotating(self)  # im = funcRotating(im , InputThreshs.angle, mode)
            if self.params.Augment.Linear.Shift.Mode:    funcShifting(self)  # im = funcShifting(im , InputThreshs.shift)
            if self.params.Augment.Linear.Shear.Mode:    funcShearing(self)  # im = funcShearing(im , InputThreshs.Shear, mode)

            self.Image = np.transpose(self.Image, self.params.WhichExperiment.Dataset.slicingInfo.slicingOrder_Reverse)

            return self.Image

    for Subjects in [params.directories.Train.Input.Subjects , params.directories.Test.Input.Subjects]:
        SubjectNames = list(Subjects.keys())

        for imInd, nameSubject in enumerate(SubjectNames):
            subject  = Subjects[nameSubject]
            for AugIx in range(params.Augment.Linear.Length):
                linearCls = linearFuncs_Class(params, nameSubject, AugIx , subject)

                print('image',imInd,'/',len(SubjectNames),'augment',AugIx,'/',params.Augment.Linear.Length , nameSubject , \
                    'sd' , params.WhichExperiment.Dataset.slicingInfo.slicingDim , 'angle' , linearCls.InputThreshs.angle)            

                def apply_OnImage():
                    imF = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')  # 'Cropped' for cropped image                        
                    Image = linearCls.main( imF.get_data() , 5)
                    smallFuncs.saveImage(Image , imF.affine , imF.header , linearCls.outDirectory.Image + '/' + subject.ImageProcessed + '.nii.gz' )            
                apply_OnImage()

                def loopOver_AllNuclei():
                    subF = [s for s in os.listdir(subject.Label.address) if 'PProcessed' in s ] 
                    for NucleusName in subF: 

                        # print(NucleusName)
                        MaskF = nib.load(subject.Label.address + '/' + NucleusName)
                        Mask = MaskF.get_data()

                        if 'Labels' not in NucleusName: Mask = smallFuncs.fixMaskMinMax(Mask,NucleusName)

                        Mask = linearCls.main( Mask , 1)  # applyLinearAugment(Mask.copy(), InputThreshs, 1)
                        smallFuncs.saveImage(Mask  , MaskF.affine , MaskF.header ,  linearCls.outDirectory.Mask  + '/' + NucleusName  )
                loopOver_AllNuclei()
            
                def loopOver_Extra_Crops():
                    subF = [s for s in os.listdir(subject.Temp.address) if '.nii.gz' in s]
                    for cropName in subF: 

                        # print(cropName)
                        MaskF = nib.load(subject.Temp.address + '/' + cropName)

                        Mask = linearCls.main( MaskF.get_data() , 1) 
                        smallFuncs.saveImage(Mask  , MaskF.affine , MaskF.header ,  linearCls.outDirectory.Temp  + '/' + cropName  )
                loopOver_Extra_Crops()                    

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


