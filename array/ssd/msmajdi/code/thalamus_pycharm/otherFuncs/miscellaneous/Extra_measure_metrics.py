import os, sys
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
from sklearn import tree
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from otherFuncs import smallFuncs, datasets
from Parameters import UserInfo, paramFunc
from otherFuncs.smallFuncs import Experiment_Folder_Search
import modelFuncs.Metrics as metrics

           

params = paramFunc.Run(UserInfo.__dict__, terminal=False)

class UserEntry():
    def __init__(self):
        self.dir_in  = ''
        self.dir_label = ''

        for en in range(len(sys.argv)):
            if sys.argv[en].lower() in ('-i','--input'):    
                self.dir_in    = os.path.abspath(sys.argv[en+1])

            elif sys.argv[en].lower() in ('-l','--label'):        
                self.dir_label = os.path.abspath(sys.argv[en+1])

            elif sys.argv[en].lower() in ('-m','--_mode'):
                self.mode      = sys.argv[en+1]
            
class measure_Metrics_cls():

    def __init__(self, dir_in = '' , dir_label = ''):

        self.dir_in    = dir_in
        self.dir_label = dir_label

    def measure_metrics(self):
            
        a = smallFuncs.Nuclei_Class().All_Nuclei()
        num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes  
        VSI       = np.zeros((num_classes-1,2))
        Dice      = np.zeros((num_classes-1,2))
        HD        = np.zeros((num_classes-1,2))

        Tag_exist = False
        for cnt, (nucleusNm , nucleiIx) in enumerate(zip(a.Names , a.Indexes)):

            Dir_prediction = self.dir_in + '/' + nucleusNm + '.nii.gz'
            Dir_Label = self.dir_label + '/Label/' + nucleusNm + '_PProcessed.nii.gz'
            if os.path.exists(Dir_Label) and os.path.exists(Dir_prediction):
                Tag_exist = True
                print(nucleusNm)
                ManualLabel = nib.load(Dir_Label).get_fdata()                                              
                prediction = nib.load(Dir_prediction).get_fdata()   
                prediction = prediction > prediction.max()/2  

                VSI[cnt,:]  = [nucleiIx , metrics.VSI_AllClasses(prediction, ManualLabel).VSI()]
                HD[cnt,:]   = [nucleiIx , metrics.HD_AllClasses(prediction, ManualLabel).HD()]
                Dice[cnt,:] = [nucleiIx , smallFuncs.mDice(prediction, ManualLabel)]

        
        if Tag_exist:
            np.savetxt( self.dir_in + '/VSI_All.txt'       ,VSI , fmt='%1.1f %1.4f')
            np.savetxt( self.dir_in + '/HD_All.txt'        ,HD , fmt='%1.1f %1.4f')
            np.savetxt( self.dir_in + '/Dice_All.txt'      ,Dice , fmt='%1.1f %1.4f')
            # np.savetxt( self.dir_in + '/Recall_All.txt'    ,Recall , fmt='%1.1f %1.4f')
            # np.savetxt( self.dir_in + '/Precision_All.txt' ,Precision , fmt='%1.1f %1.4f')  
        
                  
    def loop_All_subjects(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'case' in s]:
            print(subj , '\n')
            temp = measure_Metrics_cls(dir_in= self.dir_in + '/' + subj , dir_label= self.dir_label + '/' + subj).measure_metrics()


UI = UserEntry()

for cv in ['a','b','c','d','e','f','g','h']:
    for sd in ['sd0','sd1','sd2','2.5D_MV']:
        for x in ['ET' , 'Main']:
            UI.dir_label  = f'path-to-crossVal/{x}/{cv}' 
            UI.dir_in = f'path-to-predictions/{sd}'
            UI.mode    = 'all'

            if UI.mode == 'all':      
                measure_Metrics_cls(dir_in = UI.dir_in , dir_label = UI.dir_label).loop_All_subjects()
            else:
                measure_Metrics_cls(dir_in = UI.dir_in , dir_label = UI.dir_label).measure_metrics()