import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from otherFuncs import smallFuncs, datasets, Extra_mergingResults
import nibabel as nib
import numpy as np
from Parameters import UserInfo, paramFunc
from tqdm import tqdm

def runOneExperiment(UserInfoB):

    _, FullIndexes = smallFuncs.NucleiSelection(ind = 1,organ = 'THALAMUS')

    UserInfoB['slicingDim'] = 2
    params = paramFunc.Run(UserInfoB)

    dirSave = params.directories.Test.Result # + '_2.5D'
    smallFuncs.mkDir(dirSave + '_2.5D_MV')
    smallFuncs.mkDir(dirSave + '_2.5D_Sum')
    smallFuncs.mkDir(dirSave + '_1.5D_Sum')

    # Dirr = Fparams[2].WhichExperiment.Experiment.address + '/' + Fparams[2].WhichExperiment.SubExperiment.name
    subjects = [sj for sj in params.directories.Test.Input.Subjects if 'ERROR' not in sj]

    for sj in tqdm(subjects):
        subject = params.directories.Test.Input.Subjects[sj]

        for nucleiIx in FullIndexes:
            nucleusNm, _ = smallFuncs.NucleiSelection(ind = nucleiIx,organ = 'THALAMUS')

            if os.path.isfile(params.directories.Test.Result + '/' + sj + '/' + nucleusNm + '.nii.gz'):
                
                for slicingDim in range(3): # 
                    UserInfoB['slicingDim'] = slicingDim
                    params = paramFunc.Run(UserInfoB)   
                    im = nib.load(params.directories.Test.Result + '/' + sj + '/' + nucleusNm + '.nii.gz').get_data()[...,np.newaxis]
                    MLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
                    
                    Image = im if slicingDim == 0 else np.concatenate((Image,im),axis=3)
            else:
                print('WARNING:  test subject not found in results folder ', subject)
                break

            Image1 = Image[...,1:].sum(axis=3) >= 1
            saveImageDice(Image1, MLabel, dirSave + '_1.5D_Sum', sj, nucleusNm, nucleiIx)

            Image1 = Image.sum(axis=3) >= 2
            saveImageDice(Image1, MLabel, dirSave + '_2.5D_MV', sj, nucleusNm, nucleiIx)

            Image2 = Image.sum(axis=3) >= 1
            saveImageDice(Image2, MLabel, dirSave + '_2.5D_Sum', sj, nucleusNm, nucleiIx)

    return dirSave

def saveImageDice(Image1, MLabel, dirSave, sj, nucleusNm, nucleiIx):
    smallFuncs.saveImage( Image1 , MLabel.affine, MLabel.header, dirSave + '/' + sj + '/' + nucleusNm + '.nii.gz')
    Dice = [ nucleiIx , smallFuncs.Dice_Calculator(Image1 , MLabel.get_data()) ]
    np.savetxt(dirSave + '/' + sj + '/Dice_' + nucleusNm + '.txt' ,Dice)


AllExperimentsList = {
    1: dict(),
    # 2: dict(nucleus_Index = [6] , GPU_Index = 6 , lossFunctionIx = 2),
}

UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
UserInfoB['TestOnly'] = True
# UserInfoB['CreatingTheExperiment'] = False

dirSave = runOneExperiment(UserInfoB)

Extra_mergingResults.mergingDiceValues(dirSave.split('/subExp')[0])

os.system('bash Bash_Run')
