import os, sys
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras_run/')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import preprocess.augmentA as augmentA
import otherFuncs.smallFuncs as smallFuncs
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo

def Augment_Class():
    class rotation:
        Mode = True
        AngleMax = 6

    class shift:
        Mode = False
        ShiftMax = 10

    class shear:
        Mode = False
        ShearMax = 0

    class linearAug:
        Mode = False
        Length = 8
        Rotation = rotation()
        Shift = shift()
        Shear = shear()

    class nonlinearAug:
        Mode = False
        Length = 2
    class augment:
        Mode = False
        Linear = linearAug()
        NonLinear = nonlinearAug()

    return augment()

def Bash_AugmentNonLinear(subject , subjectRef , outputAddress): # Image , Mask , Reference , output):
    
    print('     Augment NonLinear')

    ImageOrig = subject.address       + '/' + subject.ImageProcessed + '.nii.gz'
    # MaskOrig  = subject.Label.address + '/' + subject.Label.LabelProcessed + '.nii.gz'
    ImageRef  = subjectRef.address    + '/' + subjectRef.ImageProcessed + '.nii.gz'

    OutputImage = outputAddress  + '/' + subject.ImageProcessed + '.nii.gz'
    labelAdd    = smallFuncs.mkDir(outputAddress + '/Label')
    deformationAddr = smallFuncs.mkDir(outputAddress + '/Temp/deformation')


    if not os.path.isfile(OutputImage):
        os.system("ANTS 3 -m CC[%s, %s,1,5] -t SyN[0.25] -r Gauss[3,0] -o %s -i 30x90x20 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000"%(ImageOrig , ImageRef , deformationAddr + '/test') )
        os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(ImageOrig , OutputImage , ImageOrig , deformationAddr + '/testWarp.nii.gz') )

    _, _, names = smallFuncs.NucleiSelection(ind = 1)
    for name in names:
        MaskOrig  = subject.Label.address + '/' + name + '_PProcessed.nii.gz'
        OutputMask  = labelAdd + '/' + name + '_PProcessed.nii.gz'
        if not os.path.isfile(OutputMask):
            os.system("antsApplyTransforms -d 3 -i %s -o %s -r %s -t %s"%(MaskOrig , OutputMask , MaskOrig , deformationAddr + '/testWarp.nii.gz' ) )


params = paramFunc.Run(UserInfo.__dict__, terminal=True)
params.Augment = Augment_Class()


augmentA.main_augment( params , 'Linear', 'experiment')
params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
augmentA.main_augment( params , 'NonLinear' , 'experiment')
