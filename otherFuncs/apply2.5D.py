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
    #TODO needs to be checked
    smallFuncs.saveImage( InfoSave.Image , ManualLabel.affine, ManualLabel.header, InfoSave.address  + '/' + InfoSave.nucleus.name + '.nii.gz')
    
    Label = smallFuncs.fixMaskMinMax(ManualLabel.get_data(),'ML') > 0.5

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
        # print(subject.subjectName)
        Info.subject = subject()

        a = smallFuncs.Nuclei_Class().All_Nuclei()
        for nucleusNm , nucleiIx in zip(a.Names , a.Indexes):

            if os.path.exists(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz'):
                Info.nucleus = nucleus(nucleusNm , nucleiIx)

                ix , pred3Dims = 0 , ''
                ManualLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
                for sdInfo in Info.subExperiment.multiPlanar:
                    # print(sdInfo)
                    if sdInfo.Flag and 'sd' in sdInfo.name:
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

            clf = tree.DecisionTreeClassifier(max_depth=1)
            
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
                print(subject)
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

def func_OtherMetrics_justFor_MV(Info , params):
             
    print('subExperiment:' , Info.subExperiment.name)
    Info.subExperiment.address = Info.Experiment.address + '/results/' + Info.subExperiment.name + '/'

    subjects = [s for s in params.directories.Test.Input.Subjects if 'ERROR' not in s]
    for sj in tqdm(subjects):
        subject = params.directories.Test.Input.Subjects[sj]

        a = smallFuncs.Nuclei_Class().All_Nuclei()
        num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes  
        VSI       = np.zeros((num_classes-1,2))
        # Dice      = np.zeros((num_classes-1,2))
        HD        = np.zeros((num_classes-1,2))
        # Precision = np.zeros((num_classes-1,2))
        # Recall    = np.zeros((num_classes-1,2))


        for cnt, (nucleusNm , nucleiIx) in enumerate(zip(a.Names , a.Indexes)):

            address = Info.subExperiment.address + '2.5D_MV/' + subject.subjectName + '/'

            if not os.path.exists(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz') or not os.path.isfile(address + nucleusNm + '.nii.gz'): continue
            Ref = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
            ManualLabel = Ref.get_data()
                                              
            predMV = nib.load(address + nucleusNm + '.nii.gz').get_data()                      
            VSI[cnt,:]  = [nucleiIx , metrics.VSI_AllClasses(predMV, ManualLabel).VSI()]
            HD[cnt,:]   = [nucleiIx , metrics.HD_AllClasses(predMV, ManualLabel).HD()]
            # Dice[cnt,:] = [nucleiIx , smallFuncs.mDice(predMV, ManualLabel)]

            # confusionMatrix = metrics.confusionMatrix(predMV, ManualLabel)
            # Recall[cnt,:]    = [nucleiIx , confusionMatrix.Recall]
            # Precision[cnt,:] = [nucleiIx , confusionMatrix.Precision]
            
            # np.savetxt( address + 'VSI_' + InfoSave.nucleus.name + '.txt' ,Dice , fmt='%1.1f %1.4f')
        
        np.savetxt( address + 'VSI_All.txt'       ,VSI , fmt='%1.1f %1.4f')
        np.savetxt( address + 'HD_All.txt'        ,HD , fmt='%1.1f %1.4f')
        # np.savetxt( address + 'Dice_All.txt'        ,Dice , fmt='%1.1f %1.4f')
        # np.savetxt( address + 'Recall_All.txt'    ,Recall , fmt='%1.1f %1.4f')
        # np.savetxt( address + 'Precision_All.txt' ,Precision , fmt='%1.1f %1.4f')

UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)

UserInfoB['best_network_MPlanar'] = True

UserInfoB['Model_Method'] = 'Cascade'
UserInfoB['simulation'].num_Layers = 3
# UserInfoB['simulation'].slicingDim = [2,1,0]
UserInfoB['architectureType'] = 'Res_Unet2'
UserInfoB['lossFunction_Index'] = 4
UserInfoB['Experiments'].Index = '6'
UserInfoB['copy_Thalamus'] = False
UserInfoB['TypeExperiment'] = 15
UserInfoB['simulation'].LR_Scheduler = True    
UserInfoB['tempThalamus'] = True
UserInfoB['simulation'].ReadAugments_Mode = False 
UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20

for x in ['a', 'b', 'c']:
    UserInfoB['CrossVal'].index = [x]
    params = paramFunc.Run(UserInfoB, terminal=False)
    InfoS = Experiment_Folder_Search(General_Address=params.WhichExperiment.address , Experiment_Name=params.WhichExperiment.Experiment.name , subExperiment_Name=params.WhichExperiment.SubExperiment.name)
    func_OtherMetrics(InfoS , params)                