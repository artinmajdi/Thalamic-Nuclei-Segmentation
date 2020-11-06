import os, sys

# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import nibabel as nib
import numpy as np
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
from tqdm import tqdm
from otherFuncs.smallFuncs import Experiment_Folder_Search
import matplotlib.pylab as plt
from sklearn import tree
import modelFuncs.Metrics as metrics


class infoSave:
    def __init__(self, Image=[], subject='', nucleus='', mode='', address=''):
        self.subject = subject
        self.nucleus = nucleus
        self.mode = mode
        self.address = smallFuncs.mkDir(address + self.mode + '/' + self.subject.subjectName + '/')
        self.Image = Image


class nucleus:
    def __init__(self, name='', index=0):
        self.name = name
        self.index = index


def saveImageDice(InfoSave, ManualLabel):
    # TODO needs to be checked
    smallFuncs.saveImage(InfoSave.Image, ManualLabel.affine, ManualLabel.header,
                         InfoSave.address + '/' + InfoSave.nucleus.name + '.nii.gz')

    Label = smallFuncs.fixMaskMinMax(ManualLabel.get_fdata(), 'ML') > 0.5

    Dice = np.zeros((1, 2))
    Dice[0, 0], Dice[0, 1] = InfoSave.nucleus.index, smallFuncs.mDice(InfoSave.Image, Label)
    np.savetxt(InfoSave.address + 'Dice_' + InfoSave.nucleus.name + '.txt', Dice, fmt='%1.1f %1.4f')


def func_MajorityVoting(Info, params):
    print('subExperiment:', Info.subExperiment.name)
    Info.subExperiment.address = Info.Experiment.address + '/results/' + Info.subExperiment.name + '/'

    # subjects = [s for s in os.listdir() if 'case' in s]
    subjects = [s for s in params.directories.Test.Input.Subjects if 'ERROR' not in s]
    for sj in tqdm(subjects):
        subject = params.directories.Test.Input.Subjects[sj]
        # print(subject.subjectName)
        Info.subject = subject()

        a = smallFuncs.Nuclei_Class()
        for nucleusNm, nucleiIx in zip(a.Names, a.Indexes):

            if 1:  # os.path.exists(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz'):
                Info.nucleus = nucleus(nucleusNm, nucleiIx)

                ix, pred3Dims = 0, ''
                # ManualLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
                ManualLabel = nib.load(subject.address + '/PProcessed.nii.gz')
                for sdInfo in Info.subExperiment.multiPlanar:
                    # print(sdInfo)
                    if sdInfo.Flag and 'sd' in sdInfo.name:
                        address = Info.subExperiment.address + sdInfo.name + '/' + subject.subjectName + '/' + nucleusNm + '.nii.gz'
                        if os.path.isfile(address):
                            pred = nib.load(address).get_fdata()[..., np.newaxis]
                            pred3Dims = pred if ix == 0 else np.concatenate((pred3Dims, pred), axis=3)
                            ix += 1

                if ix > 0:
                    InfoSave = infoSave(Image=pred3Dims.sum(axis=3) >= 2, subject=subject(),
                                        nucleus=nucleus(nucleusNm, nucleiIx), mode='2.5D_MV',
                                        address=Info.subExperiment.address)
                    smallFuncs.saveImage(InfoSave.Image, ManualLabel.affine, ManualLabel.header,
                                         InfoSave.address + '/' + InfoSave.nucleus.name + '.nii.gz')
                    # saveImageDice(InfoSave, ManualLabel)


def func_DecisionTree(Info, params):
    Info.subExperiment.address = Info.Experiment.address + '/results/' + Info.subExperiment.name + '/'

    a = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()
    for nucleusNm, nucleiIx in zip(a.allNames, a.Indexes):
        print(Info.subExperiment.name, nucleusNm)
        # nucleusNm, nucleiIx = '1-THALAMUS' , 1
        # a = smallFuncs.Nuclei_Class(index=6)
        # nucleusNm, nucleiIx = a.name , a.index

        Info.nucleus = nucleus(nucleusNm, nucleiIx)

        # TrainData = {}

        def training(params, Info):

            clf = tree.DecisionTreeClassifier(max_depth=1)

            for cnt, subj in enumerate(tqdm(list(params.directories.Train.Input.Subjects))):
                try:
                    subject = params.directories.Train.Input.Subjects[subj]
                    # print(cnt, len(params.directories.Train.Input.Subjects) , subject.subjectName)
                    ManualLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
                    Y = ManualLabel.get_fdata().reshape(-1)
                    X = np.zeros((np.prod(ManualLabel.shape), 3))

                    for ix, sd in enumerate(['sd0', 'sd1', 'sd2']):
                        address = Info.subExperiment.address + sd + '/TrainData_Output/' + subject.subjectName + '/' + nucleusNm + '.nii.gz'
                        X[:, ix] = nib.load(address).get_fdata().reshape(-1)

                    # if cnt == 0:
                    #     TrainData = X.copy()
                    #     TrainLabel = Y.copy()
                    # else:
                    #     TrainData  = np.concatenate((TrainData,X),axis=0)
                    #     TrainLabel = np.concatenate((TrainLabel,Y),axis=0)

                    clf = clf.fit(X, Y > 0)
                except:
                    print('crashed', subj)

            # clf = clf.fit(TrainData,TrainLabel>0)

            return clf

        def testing(params, Info, clf):
            for cnt, subj in enumerate(list(params.directories.Test.Input.Subjects)):

                subject = params.directories.Test.Input.Subjects[subj]
                print(subject)
                print(cnt, len(params.directories.Test.Input.Subjects), subject.subjectName)
                ManualLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
                Yt = ManualLabel.get_fdata().reshape(-1)
                Xt = np.zeros((np.prod(ManualLabel.shape), 3))

                for ix, sd in enumerate(['sd0', 'sd1', 'sd2']):
                    address = Info.subExperiment.address + sd + '/' + subject.subjectName + '/' + nucleusNm + '.nii.gz'
                    Xt[:, ix] = nib.load(address).get_fdata().reshape(-1)

                out = clf.predict(Xt)

                pred = out.reshape(ManualLabel.shape)

                InfoSave = infoSave(Image=pred, subject=subject(), nucleus=nucleus(nucleusNm, nucleiIx), mode='DT',
                                    address=Info.subExperiment.address)
                saveImageDice(InfoSave, ManualLabel)

        clf = training(params, Info)
        testing(params, Info, clf)


def func_OtherMetrics_justFor_MV(Info, params):
    print('subExperiment:', Info.subExperiment.name)
    Info.subExperiment.address = Info.Experiment.address + '/results/' + Info.subExperiment.name + '/'

    subjects = [s for s in params.directories.Test.Input.Subjects if 'ERROR' not in s]
    for sj in tqdm(subjects):
        subject = params.directories.Test.Input.Subjects[sj]

        a = smallFuncs.Nuclei_Class().All_Nuclei()
        num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes
        VSI = np.zeros((num_classes - 1, 2))
        Dice = np.zeros((num_classes - 1, 2))
        HD = np.zeros((num_classes - 1, 2))
        Volumes = np.zeros((num_classes - 1, 2))
        # Precision = np.zeros((num_classes-1,2))
        # Recall    = np.zeros((num_classes-1,2))

        for cnt, (nucleusNm, nucleiIx) in enumerate(zip(a.Names, a.Indexes)):

            address = Info.subExperiment.address + '2.5D_MV/' + subject.subjectName + '/'

            if not os.path.exists(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz') or not os.path.isfile(
                address + nucleusNm + '.nii.gz'): continue
            Ref = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
            ManualLabel = Ref.get_fdata()

            predMV = nib.load(address + nucleusNm + '.nii.gz').get_fdata()
            VSI[cnt, :] = [nucleiIx, metrics.VSI_AllClasses(predMV, ManualLabel).VSI()]
            HD[cnt, :] = [nucleiIx, metrics.HD_AllClasses(predMV, ManualLabel).HD()]
            Dice[cnt, :] = [nucleiIx, smallFuncs.mDice(predMV, ManualLabel)]
            Volumes[cnt, :] = [nucleiIx, predMV.sum()]

            # confusionMatrix = metrics.confusionMatrix(predMV, ManualLabel)
            # Recall[cnt,:]    = [nucleiIx , confusionMatrix.Recall]
            # Precision[cnt,:] = [nucleiIx , confusionMatrix.Precision]

            # np.savetxt( address + 'VSI_' + InfoSave.nucleus.name + '.txt' ,DICE , fmt='%1.1f %1.4f')

        np.savetxt(address + 'VSI_All.txt', VSI, fmt='%1.1f %1.4f')
        np.savetxt(address + 'HD_All.txt', HD, fmt='%1.1f %1.4f')
        np.savetxt(address + 'Dice_All.txt', Dice, fmt='%1.1f %1.4f')
        np.savetxt(address + 'Volumes_All.txt', Volumes, fmt='%1.1f %1.4f')
        # np.savetxt( address + 'Recall_All.txt'    ,Recall , fmt='%1.1f %1.4f')
        # np.savetxt( address + 'Precision_All.txt' ,Precision , fmt='%1.1f %1.4f')


def func_AllMetrics_UserDirectory(Dir, params):
    Dir_ManualLabels = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset2_with_Manual_Labels/full_Image/freesurfer/ManualLabels2_uncropped'
    subjects = [s for s in os.listdir(Dir) if 'case' in s]

    for subjectName in tqdm(subjects):
        # subject = params.directories.Test.Input.Subjects[sj]

        # try:

        a = smallFuncs.Nuclei_Class().All_Nuclei()
        num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes
        VSI = np.zeros((num_classes - 1, 2))
        Dice = np.zeros((num_classes - 1, 2))
        HD = np.zeros((num_classes - 1, 2))
        # Precision = np.zeros((num_classes-1,2))
        # Recall    = np.zeros((num_classes-1,2))

        for cnt, (nucleusNm, nucleiIx) in enumerate(zip(a.Names, a.Indexes)):
            # address = Info.subExperiment.address + '2.5D_MV/' + subject.subjectName + '/'
            address = Dir + '/' + subjectName + '/'

            # if not os.path.exists(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz') or not os.path.isfile(address + nucleusNm + '.nii.gz'): continue

            Flag = os.path.exists(address + 'Prediction/' + nucleusNm + '.nii.gz')
            # if os.path.exists(address + 'Label_uncropped/' + nucleusNm + '.nii.gz') and Flag:
            #     manual_dir = address + 'Label_uncropped/' + nucleusNm + '.nii.gz'

            manual_dir = Dir_ManualLabels + '/' + subjectName + '/Label/' + nucleusNm + '.nii.gz'
            # if os.path.exists(address + 'Label/' + nucleusNm + '.nii.gz') and Flag:
            #     manual_dir = address + 'Label/' + nucleusNm + '.nii.gz'                
            # else:
            #     continue

            print(subjectName, nucleusNm)
            ManualLabel = nib.load(manual_dir).get_fdata()
            prediction = nib.load(address + 'Prediction/' + nucleusNm + '.nii.gz').get_fdata()
            prediction = prediction > prediction.max() / 2

            VSI[cnt, :] = [nucleiIx, metrics.VSI_AllClasses(prediction, ManualLabel).VSI()]
            HD[cnt, :] = [nucleiIx, metrics.HD_AllClasses(prediction, ManualLabel).HD()]
            Dice[cnt, :] = [nucleiIx, smallFuncs.mDice(prediction, ManualLabel)]

            # confusionMatrix = metrics.confusionMatrix(predMV, ManualLabel)
            # Recall[cnt,:]    = [nucleiIx , confusionMatrix.Recall]
            # Precision[cnt,:] = [nucleiIx , confusionMatrix.Precision]

        np.savetxt(address + 'VSI_All.txt', VSI, fmt='%1.1f %1.4f')
        np.savetxt(address + 'HD_All.txt', HD, fmt='%1.1f %1.4f')
        np.savetxt(address + 'Dice_All.txt', Dice, fmt='%1.1f %1.4f')
        # np.savetxt( address + 'Recall_All.txt'    ,Recall , fmt='%1.1f %1.4f')
        # np.savetxt( address + 'Precision_All.txt' ,Precision , fmt='%1.1f %1.4f')

        # except Exception as e:
        #     print(subjectName, e)


UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)

UserInfoB['TypeExperiment'] = 11
UserInfoB['simulation'].lossFunction_Index = 4
UserInfoB['Experiments'].Index = '6'
UserInfoB['copy_Thalamus'] = False
UserInfoB['simulation'].batch_size = 50
UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20
UserInfoB['simulation'].FCN1_NLayers = 0
UserInfoB['simulation'].FCN2_NLayers = 0
UserInfoB['simulation'].FCN_FeatureMaps = 0
UserInfoB['simulation'].LR_Scheduler = False
UserInfoB['Experiments'].Tag = 'BC_CSFn'

for x in ['a', 'b', 'c', 'd']:
    params = paramFunc.Run(UserInfoB, terminal=False)
    InfoS = Experiment_Folder_Search(General_Address=params.WhichExperiment.address,
                                     Experiment_Name=params.WhichExperiment.Experiment.name,
                                     subExperiment_Name=params.WhichExperiment.SubExperiment.name)
    # func_MajorityVoting(InfoS , params)
    func_OtherMetrics_justFor_MV(InfoS, params)

    # params = paramFunc.Run(UserInfo.__dict__, terminal=False)
    # Dir = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset2_with_Manual_Labels/full_Image/freesurfer/step2_freesurfer/Done/step2_resliced'
    # func_AllMetrics_UserDirectory(Dir , params)
