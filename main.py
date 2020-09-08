import os
import sys

sys.path.append(os.path.dirname(__file__))
import otherFuncs.smallFuncs as smallFuncs
from otherFuncs import datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess
from keras import backend as K
import nibabel as nib
import numpy as np

UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
UserInfoB['simulation'] = UserInfoB['simulation']()
K = smallFuncs.gpuSetting(UserInfoB['simulation'].GPU_Index)


def main(UserInfoB):
    def Save_AllNuclei_inOne(Directory, mode='_PProcessed'):
        """ Saving all of the predicted nuclei into one nifti image

        Args:
            Directory (str): The path to all predicted nuclei
            mode (str, optional): Optional tag that can be added to the predicted nuclei names. Defaults to '_PProcessed'.
        """

        mask = []

        # Looping through all nuclei
        for cnt in (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14):

            name = smallFuncs.Nuclei_Class(index=cnt).name
            dirr = Directory + '/' + name + mode + '.nii.gz'
            if os.path.isfile(dirr):
                if cnt == 1:
                    # loading thalamus mask for the purpose of using its affine matrices & header
                    assert os.path.isfile(dirr), 'Thalamus mask does not exist'
                    thalamus_mask = nib.load(dirr)

                else:
                    # saving the nuclei into one mask
                    msk = nib.load(dirr).get_data()
                    if not mask:
                        # saving the first nucleus (2-AV)
                        mask = cnt * msk
                    else:
                        # saving the remaining nuclei, while randomly assigning a label from the labels 
                        # that exist in the overlapping area
                        mask_temp = mask.copy()
                        mask_temp[msk == 0] = 0
                        x = np.where(mask_temp > 0)
                        if x[0].shape[0] > 0:
                            fg = np.random.randn(x[0].shape[0])
                            fg1, fg2 = fg >= 0, fg < 0
                            mask[x[0][fg1], x[1][fg1], x[2][fg1]] = 0
                            msk[x[0][fg2], x[1][fg2], x[2][fg2]] = 0

                        mask += cnt * msk

                        # Saving the final multi-label segmentaion mask as a nifti image
        smallFuncs.saveImage(mask, thalamus_mask.affine, thalamus_mask.header, Directory + '/AllLabels.nii.gz')

    def running_main(UserInfoB):
        """ Running the network on left and/or right thalamus

        Args:
            UserInfoB: User Inputs
        """

        def Run(UserInfoB):
            """ Loading the dataset & running the network on the assigned slicing orientation & nuclei

            Args:
                UserInfoB: User Inputs
            """

            params = paramFunc.Run(UserInfoB, terminal=True)
            print('\n', params.WhichExperiment.Nucleus.name, 'SD: ' + str(UserInfoB['simulation'].slicingDim),
                  'GPU: ' + str(UserInfoB['simulation'].GPU_Index), '\n')

            # Loading the dataset
            Data, params = datasets.loadDataset(params)

            # Running the training/testing network
            choosingModel.check_Run(params, Data)

            # clearing the gpu session
            K.clear_session()

        def merge_results_and_apply_25D(UserInfoB):
            """ Merging the sagittal, Coronal, and axial networks prediction masks using 2.5D majority voting
            """

            params = paramFunc.Run(UserInfoB, terminal=True)
            smallFuncs.apply_MajorityVoting(params)

        def predict_thalamus_for_sd0(UserI):
            """ Due to the existense of both left and right thalamus in the cropped nifti image while lacking the
                manual labels for the right thalamus, during the sagittal network process, to predict the whole thalamus
                all 3D volumes will be sampled in the coronal direction instead of sagittal
            """

            # Predicting the whole thalamus in the coronal orientation
            UserI['simulation'].slicingDim = [2]
            UserI['simulation'].nucleus_Index = [1]
            UserI['simulation'].Use_Coronal_Thalamus_InSagittal = True
            Run(UserI)

            # Predicting the remaining of nuclei in the sagittal orientation
            UserI['simulation'].slicingDim = [0]
            UserI['simulation'].nucleus_Index = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            Run(UserI)

        def predict_multi_thalamus(UserI):
            """ Running the two consecutive networks in the cascaded algorithm for axial & coronal orientations
            """

            # Running the 1st network of cascaded algorithm: Predicting whole thalamus 
            UserI['simulation'].nucleus_Index = [1]
            Run(UserI)

            # Running the 2nd network of cascaded algorithm: Predicting the remaiing of nuclei 
            # after cropping the input image & its nuclei using predicted whole thalamus bounding box
            UserI['simulation'].nucleus_Index = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            Run(UserI)

        # Running the sagittal network
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 40  # Number of feature maps in the first layer of Resnet
        UserInfoB['simulation'].slicingDim = [0]  # Sagittal Orientation
        UserInfoB['simulation'].nucleus_Index = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        predict_thalamus_for_sd0(UserInfoB)

        # Running the axial network
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 30  # Number of feature maps in the first layer of Resnet
        UserInfoB['simulation'].slicingDim = [1]  # Axial Orientation
        UserInfoB['simulation'].nucleus_Index = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        predict_multi_thalamus(UserInfoB)

        # Running the coronal network
        UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20  # Number of feature maps in the first layer of Resnet
        UserInfoB['simulation'].slicingDim = [2]  # Coronal Orientation
        UserInfoB['simulation'].nucleus_Index = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        # UserInfoB['simulation'].Use_Coronal_Thalamus_InSagittal = False
        predict_multi_thalamus(UserInfoB)

        merge_results_and_apply_25D(UserInfoB)

    def run_Left(UserInfoB):
        """ running the network on left thalamus """

        UserInfoB['thalamic_side'].active_side = 'left'
        running_main(UserInfoB)

        # Saving the multi label nifti image consisting of all predicted labels from 2-AV to 14-MTT 
        params = paramFunc.Run(UserInfoB, terminal=True)
        for subj in params.directories.Test.Input.Subjects.values():
            Save_AllNuclei_inOne(subj.address + '/left/2.5D_MV', mode='')

    def run_Right(UserInfoB):
        """ running the network on right thalamus """

        def flip_inputs(params):
            print('Flip L-R the image & its nuclei')

            subjects = params.directories.Test.Input.Subjects.copy()
            code_address = params.WhichExperiment.Experiment.code_address + '/otherFuncs/flip_inputs.py'
            subjects.update(params.directories.Train.Input.Subjects)

            for subj in subjects.values():
                os.system("cd {0};python {1} -i {0}/PProcessed.nii.gz -o {0}/PProcessed.nii.gz;".format(subj.address,
                                                                                                        code_address))
                os.system("cd {0};mv {0}/PProcessed.nii.gz {0}/flipped_PProcessed.nii.gz;".format(subj.address))

        def unflip_inputs(params):
            print('Reverse Flip L-R the flipped image & its nuclei')

            subjects = params.directories.Test.Input.Subjects.copy()
            code_address = params.WhichExperiment.Experiment.code_address + '/otherFuncs/flip_inputs.py'
            subjects.update(params.directories.Train.Input.Subjects)

            for subj in subjects.values():
                os.system(
                    "cd {0};for n in flipped_PProcessed.nii.gz right/*/*.nii.gz; do python {1} -i {0}/$n -o {0}/$n; done".format(
                        subj.address, code_address))
                os.system("cd {0};mv {0}/flipped_PProcessed.nii.gz {0}/PProcessed.nii.gz;".format(subj.address))
                # os.system("cd %s;for n in *.nii.gz ; do mv $n ${n#*_} ; done"%(subj.address))  # ${a#*_}   

        UserInfoB['thalamic_side'].active_side = 'right'
        params = paramFunc.Run(UserInfoB, terminal=True)

        # Flipping the data
        flip_inputs(params)

        # Running the trained network on right thalamus
        running_main(UserInfoB)

        # Flipping the data back to its original orientation
        unflip_inputs(params)

        # Looping through subjects: Saving the multi label nifti image consisting of all predicted labels from 2-AV to 14-MTT 
        for subj in params.directories.Test.Input.Subjects.values():
            Save_AllNuclei_inOne(subj.address + '/right/2.5D_MV', mode='')

    def merging_left_right_labels(UserInfoB):
        params = paramFunc.Run(UserInfoB, terminal=True)
        for subj in params.directories.Test.Input.Subjects.values():
            load_side = lambda side: nib.load(subj.address + '/' + side + '/2.5D_MV/AllLabels.nii.gz')
            left, right = load_side('left'), load_side('right')
            smallFuncs.saveImage(Image=left.get_data() + right.get_data(), Affine=left.affine, Header=left.header,
                                 outDirectory=subj.address + '/left/AllLabels_Left_and_Right.nii.gz')

    applyPreprocess.main(paramFunc.Run(UserInfoB, terminal=True))

    TS = UserInfoB['thalamic_side']()
    if TS.left:              run_Left(UserInfoB)
    if TS.right:             run_Right(UserInfoB)
    if TS.left and TS.right: merging_left_right_labels(UserInfoB)


if __name__ == '__main__':
    main(UserInfoB)
