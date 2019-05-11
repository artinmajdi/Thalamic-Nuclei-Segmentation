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

class infoSave:
    def __init__(self, Image = [] , subject = '', nucleus='' , mode = '' , address=''):                
        self.subject = subject
        self.nucleus = nucleus
        self.mode = mode
        self.address = smallFuncs.mkDir(address + self.mode + '/' + self.subject.subjectName + '/')
        self.Image = Image

class nucleus:
    def __init__(self,name='', index=0):
        self.name = name
        self.index = index        

def saveImageDice(InfoSave, ManualLabel):
    smallFuncs.saveImage( InfoSave.Image , ManualLabel.affine, ManualLabel.header, InfoSave.address  + InfoSave.nucleus.name + '.nii.gz')
    
    Label = smallFuncs.fixMaskMinMax(ManualLabel.get_data()) > 0.5

    Dice = np.zeros((1,2))
    Dice[0,0] , Dice[0,1] = InfoSave.nucleus.index , smallFuncs.mDice(InfoSave.Image  , Label)
    np.savetxt( InfoSave.address+ 'Dice_' + InfoSave.nucleus.name + '.txt' ,Dice , fmt='%1.1f %1.4f')
            
def func_MajorityVoting(Info , params):
             
    print('subExperiment:' , Info.subExperiment.name)
    Info.subExperiment.address = Info.Experiment.address + '/results/' + Info.subExperiment.name + '/'


    # subjects = [s for s in os.listdir() if 'vimp' in s]
    subjects = [s for s in params.directories.Test.Input.Subjects if 'ERROR' not in s]
    for sj in tqdm(subjects):
        subject = params.directories.Test.Input.Subjects[sj]
        Info.subject = subject()

        a = smallFuncs.Nuclei_Class().All_Nuclei()
        for nucleusNm , nucleiIx in zip(a.Names , a.Indexes):

            if os.path.exists(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz'):
                Info.nucleus = nucleus(nucleusNm , nucleiIx)

                ix , pred3Dims = 0 , ''
                ManualLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
                for sdInfo in Info.subExperiment.multiPlanar:
                    if sdInfo.mode and 'sd' in sdInfo.name:
                        address = Info.subExperiment.address + sdInfo.name + '/' + subject.subjectName + '/' + nucleusNm + '.nii.gz'
                        if os.path.isfile(address):
                            
                            pred = nib.load(address).get_data()[...,np.newaxis]                                                
                            pred3Dims = pred if ix == 0 else np.concatenate((pred3Dims,pred),axis=3)
                            ix += 1

                if ix > 0:
                    InfoSave = infoSave(Image = pred3Dims.sum(axis=3) >= 2 , subject = subject() , nucleus=nucleus(nucleusNm , nucleiIx) , mode = '2.5D_MV' , address=Info.subExperiment.address) 
                    saveImageDice(InfoSave, ManualLabel)

def func_DecisionTree(Info , params):

    Info.subExperiment.address = Info.Experiment.address + '/results/' + Info.subExperiment.name + '/'
   
    a = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()
    for nucleusNm , nucleiIx in zip(a.Names , a.Indexes):
        print(Info.subExperiment.name , nucleusNm)
        # nucleusNm, nucleiIx = '1-THALAMUS' , 1
        # a = smallFuncs.Nuclei_Class(index=6)
        # nucleusNm, nucleiIx = a.name , a.index

        Info.nucleus = nucleus(nucleusNm , nucleiIx)

        # TrainData = {}
        
        def training(params , Info):

            clf = tree.DecisionTreeClassifier(max_depth=3)
            
            for cnt , subj in enumerate(tqdm(list(params.directories.Train.Input.Subjects))):
                try: 
                    subject = params.directories.Train.Input.Subjects[subj]
                    # print(cnt, len(params.directories.Train.Input.Subjects) , subject.subjectName)
                    ManualLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')        
                    Y = ManualLabel.get_data().reshape(-1)
                    X = np.zeros((np.prod(ManualLabel.shape),3))
                    
                    for ix , sd in enumerate(['sd0' , 'sd1' , 'sd2']):
                        address = Info.subExperiment.address + sd + '/TrainData_Output/' + subject.subjectName + '/' + nucleusNm + '.nii.gz'                           
                        X[:,ix] = nib.load(address).get_data().reshape(-1)
                        
                    # if cnt == 0:
                    #     TrainData = X.copy()
                    #     TrainLabel = Y.copy()
                    # else:
                    #     TrainData  = np.concatenate((TrainData,X),axis=0)
                    #     TrainLabel = np.concatenate((TrainLabel,Y),axis=0)

                    clf = clf.fit(X,Y>0)
                except:
                    print('crashed',subj)

            # clf = clf.fit(TrainData,TrainLabel>0)

            return clf

        def testing(params, Info , clf):
            for cnt , subj in enumerate(list(params.directories.Test.Input.Subjects)):

                subject = params.directories.Test.Input.Subjects[subj]
                print(cnt, len(params.directories.Test.Input.Subjects) , subject.subjectName)
                ManualLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')        
                Yt = ManualLabel.get_data().reshape(-1)
                Xt = np.zeros((np.prod(ManualLabel.shape),3))
                
                for ix , sd in enumerate(['sd0' , 'sd1' , 'sd2']):
                    address = Info.subExperiment.address + sd + '/' + subject.subjectName + '/' + nucleusNm + '.nii.gz'                           
                    Xt[:,ix] = nib.load(address).get_data().reshape(-1)
                    
                out = clf.predict(Xt)

                pred = out.reshape(ManualLabel.shape)

                InfoSave = infoSave(Image = pred , subject = subject() , nucleus=nucleus(nucleusNm , nucleiIx) , mode = 'DT' , address=Info.subExperiment.address) 
                saveImageDice(InfoSave, ManualLabel)

        clf = training(params , Info)
        testing(params, Info , clf)



UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)

# for subExperiment in ['sE11_mUnet_FM20_DO0.3_Main_InitFrom_Th_CV_a' , 'sE11_mUnet_FM20_DO0.3_Main_PlustET_InitFrom_Th_CV_a'] : #UserInfoB['Model_Method'] in ['Cascade' , 'HCascade']:
params = paramFunc.Run(UserInfoB, terminal=False)

InfoS = Experiment_Folder_Search(General_Address=params.WhichExperiment.address , Experiment_Name=params.WhichExperiment.Experiment.name , subExperiment_Name=params.WhichExperiment.SubExperiment.name)
# func_MajorityVoting(InfoS , params)
func_DecisionTree(InfoS , params)
#    print(subExperiment)
