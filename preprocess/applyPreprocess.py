# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
from preprocess import augmentA, BashCallingFunctionsA, normalizeA, croppingA

# TODO  check 3T 7T dimension and interpolation
# TODO check image format and convert to nifti

#! mode: 1: on train & test folders in the experiment
#! mode: 2: on individual image
def main(params, mode):

    # params = smallFuncs.inputNamesCheck(params, mode)
    if 'experiment' in mode:
        params = apply_On_Experiment(params)
    else:
        subject = []
        params = apply_On_Individual(params,subject)

    return params

# TODO need to fix the "BashCallingFunctionsA" function to count for situations when we only want to apply the function on one case
def apply_On_Individual(params,subject):

    BashCallingFunctionsA.BiasCorrection( subject , params)
    BashCallingFunctionsA.RigidRegistration( subject , params.WhichExperiment.HardParams.Template , params.preprocess)
    BashCallingFunctionsA.Bash_Cropping( subject , params)

    return params

def apply_On_Experiment(params):

    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)

    for mode in ['train','test']:

        if params.preprocess.TestOnly and 'train' in mode:
            continue

        dirr = params.directories.Train if mode == 'train' else params.directories.Test
        for ind, sj in enumerate(dirr.Input.Subjects):
            subject = dirr.Input.Subjects[sj]
            print(mode.upper(), 'BiasCorrection:' , sj , str(ind) + '/' + str(len(dirr.Input.Subjects)))
            BashCallingFunctionsA.BiasCorrection( subject , params)

    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)    
    for mode in ['train','test']:

        dirr = params.directories.Train if mode == 'train' else params.directories.Test
        for ind, sj in enumerate(dirr.Input.Subjects) :
            subject = dirr.Input.Subjects[sj]
            if 'Aug' not in sj:
                print('\n',mode.upper(), sj , str(ind) + '/' + str(len(dirr.Input.Subjects)),'------------')
                print('    - RigidRegistration: ')
                BashCallingFunctionsA.RigidRegistration( subject , params.WhichExperiment.HardParams.Template , params.preprocess)
            else:
                print('----')

            print('    - Cropping: ')
            croppingA.main(subject , params)
            # BashCallingFunctionsA.Bash_Cropping( subject , params)

    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    augmentA.main_augment( params , 'Linear' , 'experiment')
    params.directories = smallFuncs.search_ExperimentDirectory(params.WhichExperiment)
    augmentA.main_augment( params , 'NonLinear' , 'experiment')

    return params
