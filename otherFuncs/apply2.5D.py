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

def runOneExperiment(UserInfoB):

    _, FullIndexes,_ = smallFuncs.NucleiSelection(ind = 1)

    UserInfoB['simulation'].slicingDim = [2]
    params = paramFunc.Run(UserInfoB, terminal=False)

    dirSave = params.directories.Test.Result 
    dirSave = dirSave.replace('_sd2','')
    smallFuncs.mkDir(dirSave + '_2.5D_MV')
    smallFuncs.mkDir(dirSave + '_2.5D_Sum')
    smallFuncs.mkDir(dirSave + '_1.5D_Sum')

    # Dirr = Fparams[2].WhichExperiment.Experiment.address + '/' + Fparams[2].WhichExperiment.SubExperiment.name
    subjects = [sj for sj in params.directories.Test.Input.Subjects if 'ERROR' not in sj]

    for sj in tqdm(subjects):
        subject = params.directories.Test.Input.Subjects[sj]
        
        for nucleiIx in tuple(FullIndexes) + (1.1, 1.2, 1.3):
            
            nucleusNm, _,_ = smallFuncs.NucleiSelection(ind = nucleiIx)            
            if os.path.isfile(params.directories.Test.Result + '/' + sj + '/' + nucleusNm + '.nii.gz'):
                
                print(subject.subjectName, nucleusNm)    
                try:            
                    for slicingDim in range(3):
                        UserInfoB['simulation'].slicingDim = [slicingDim]
                        params = paramFunc.Run(UserInfoB, terminal=False)   
                        pred = nib.load(params.directories.Test.Result + '/' + sj + '/' + nucleusNm + '.nii.gz').get_data()[...,np.newaxis]
                        ManualLabel = nib.load(subject.Label.address + '/' + nucleusNm + '_PProcessed.nii.gz')
                        
                        pred3Dims = pred if slicingDim == 0 else np.concatenate((pred3Dims,pred),axis=3)

                    Image1 = pred3Dims[...,1:].sum(axis=3) >= 1
                    saveImageDice(Image1, ManualLabel, dirSave + '_1.5D_Sum', sj, nucleusNm, nucleiIx)

                    Image1 = pred3Dims.sum(axis=3) >= 2
                    saveImageDice(Image1, ManualLabel, dirSave + '_2.5D_MV', sj, nucleusNm, nucleiIx)

                    Image2 = pred3Dims.sum(axis=3) >= 1
                    saveImageDice(Image2, ManualLabel, dirSave + '_2.5D_Sum', sj, nucleusNm, nucleiIx)
                except:
                    print('failed')
                
    return dirSave

def saveImageDice(Image1, ManualLabel, dirSave, sj, nucleusNm, nucleiIx):
    smallFuncs.saveImage( Image1 , ManualLabel.affine, ManualLabel.header, dirSave + '/' + sj + '/' + nucleusNm + '.nii.gz')
    Dice = [ nucleiIx , smallFuncs.mDice(Image1 , ManualLabel.get_data()) ]
    np.savetxt(dirSave + '/' + sj + '/Dice_' + nucleusNm + '.txt' ,Dice)


UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)

dirSave = runOneExperiment(UserInfoB)

# Extra_mergingResults()
