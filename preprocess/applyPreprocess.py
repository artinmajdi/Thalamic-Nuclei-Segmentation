# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
from preprocess import augmentA, BashCallingFunctionsA, normalizeA, croppingA

# TODO  check 3T 7T dimension and interpolation
# TODO check image format and convert to nifti

#! mode: 1: on train & test folders in the experiment
#! mode: 2: on individual image
def main(params, mode):

    def loopOverSubjects_PreProcessing(params, Mode):
        
        class Info:
            mode = Mode
            dirr = params.directories.Train if Mode == 'train' else params.directories.Test
            Length = len(dirr.Input.Subjects)
            subjectName = ''
            ind = ''
            Subjects = dirr.Input.Subjects

        for Info.ind, Info.subjectName in enumerate(Info.Subjects): apply_On_Individual(params, Info)

    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    if not params.preprocess.TestOnly: loopOverSubjects_PreProcessing(params, 'train')

    loopOverSubjects_PreProcessing(params, 'test')

def apply_On_Individual(params,Info):

    subject = Info.Subjects[Info.subjectName]

    print( '(' + str(Info.ind) + '/'+str(Info.Length) + ')' , Info.mode, Info.subjectName)

    BashCallingFunctionsA.BiasCorrection( subject , params)

    if 'Aug' not in Info.subjectName: 
        BashCallingFunctionsA.RigidRegistration( subject , params.WhichExperiment.HardParams.Template , params.preprocess)

    croppingA.main(subject , params)

    return params

def apply_Augmentation(params):
    augmentA.main_augment( params , 'Linear' , 'experiment')
    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    augmentA.main_augment( params , 'NonLinear' , 'experiment')



