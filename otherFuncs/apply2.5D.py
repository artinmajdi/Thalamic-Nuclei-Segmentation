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
def runOneExperiment(Info , params):

    def saveImageDice(InfoSave, ManualLabel):
        smallFuncs.saveImage( InfoSave.Image , ManualLabel.affine, ManualLabel.header, InfoSave.address  + InfoSave.nucleus.name + '.nii.gz')
        Dice = np.zeros((1,2))
        Dice[0,0] , Dice[0,1] = InfoSave.nucleus.index , smallFuncs.mDice(InfoSave.Image  , ManualLabel.get_data())
        np.savetxt( InfoSave.address+ 'Dice_' + InfoSave.nucleus.name + '.txt' ,Dice , fmt='%1.1f %1.4f')
             

    Info.subExperiment.address = Info.Experiment.address + '/results/' + Info.subExperiment.name + '/'
    class nucleus:
        def __init__(self,name='', index=0):
            self.name = name
            self.index = index
            
    class infoSave:
        def __init__(self, Image = [] , subject = '', nucleus='' , mode = '' , address=''):                
            self.subject = subject
            self.nucleus = nucleus
            self.mode = mode
            self.address = smallFuncs.mkDir(address + self.mode + '/' + self.subject.subjectName + '/')
            self.Image = Image


    subjects = [sj for sj in params.directories.Test.Input.Subjects if 'ERROR' not in sj]
    for sj in tqdm(subjects):
        subject = params.directories.Test.Input.Subjects[sj]
        Info.subject = subject()

        a = smallFuncs.NucleiIndex().All_Nuclei
        for nucleusNm , nucleiIx in zip(a.Names , a.Indexes):
            Info.nucleus = nucleus(nucleusNm , nucleiIx)

            ix , pred3Dims = 0 , ''
            ManualLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
            for sdInfo in Info.subExperiment.multiPlanar:
                if sdInfo.mode:
                    address = Info.subExperiment.address + sdInfo.name + '/' + sj + '/' + nucleusNm + '.nii.gz'
                    if os.path.isfile(address):
                        
                        pred = nib.load(address).get_data()[...,np.newaxis]                                                
                        pred3Dims = pred if ix == 0 else np.concatenate((pred3Dims,pred),axis=3)
                        ix += 1

            if ix > 0:
                InfoSave = infoSave(Image = pred3Dims[...,1:].sum(axis=3) >= 1 , subject = subject() , nucleus=nucleus(nucleusNm , nucleiIx) , mode = '1.5D_Sum' , address=Info.subExperiment.address) 
                saveImageDice(InfoSave, ManualLabel)

                InfoSave = infoSave(Image = pred3Dims.sum(axis=3) >= 2 , subject = subject() , nucleus=nucleus(nucleusNm , nucleiIx) , mode = '2.5D_MV' , address=Info.subExperiment.address) 
                saveImageDice(InfoSave, ManualLabel)

                InfoSave = infoSave(Image = pred3Dims.sum(axis=3) >= 1  , subject = subject() , nucleus=nucleus(nucleusNm , nucleiIx) , mode = '2.5D_Sum' , address=Info.subExperiment.address) 
                saveImageDice(InfoSave, ManualLabel)

            
UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)

for UserInfoB['Model_Method'] in ['Cascade' , 'HCascade']:
    params = paramFunc.Run(UserInfoB, terminal=False)
    InfoS = Experiment_Folder_Search(General_Address=params.WhichExperiment.address , Experiment_Name=params.WhichExperiment.Experiment.name , subExperiment_Name=params.WhichExperiment.SubExperiment.name)
    runOneExperiment(InfoS , params)
    print(params.WhichExperiment.SubExperiment.name)
