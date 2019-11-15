import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append('/code')
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

           

params = paramFunc.Run(UserInfo.__dict__, terminal=False)

class UserEntry():
    def __init__(self):
        self.dir_in  = ''
        self.dir_label = ''

        for en in range(len(sys.argv)):
            if sys.argv[en].lower() in ('-i','--input'):    self.dir_in    = os.getcwd() + '/' + sys.argv[en+1] if '/array/ssd' not in sys.argv[en+1] else sys.argv[en+1] 
            elif sys.argv[en].lower() in ('-label'):        self.dir_label = os.getcwd() + '/' + sys.argv[en+1] if '/array/ssd' not in sys.argv[en+1] else sys.argv[en+1] 
            elif sys.argv[en].lower() in ('-m','--mode'):   self.mode      = sys.argv[en+1]
            
class measure_Metrics_cls():

    def __init__(self, dir_in = '' , dir_label = ''):

        self.dir_in    = dir_in
        self.dir_label = dir_label

    def measure_metrics(self):

        # Dir_ManualLabels = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset2_with_Manual_Labels/full_Image/freesurfer/ManualLabels2_uncropped'        
            
        a = smallFuncs.Nuclei_Class().All_Nuclei()
        num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes  
        VSI       = np.zeros((num_classes-1,2))
        Dice      = np.zeros((num_classes-1,2))
        HD        = np.zeros((num_classes-1,2))
        # Precision = np.zeros((num_classes-1,2))
        # Recall    = np.zeros((num_classes-1,2))


        for cnt, (nucleusNm , nucleiIx) in enumerate(zip(a.Names , a.Indexes)):

            
            if os.path.exists(self.dir_in + '/Label/' + nucleusNm + '.nii.gz'):
                print(nucleusNm)
                ManualLabel = nib.load(self.dir_label + '/Label/' + nucleusNm + '_PProcessed.nii.gz').get_data()                                              
                prediction = nib.load(self.dir_in + '/' + nucleusNm + '.nii.gz').get_data()   
                prediction = prediction > prediction.max()/2  

                VSI[cnt,:]  = [nucleiIx , metrics.VSI_AllClasses(prediction, ManualLabel).VSI()]
                HD[cnt,:]   = [nucleiIx , metrics.HD_AllClasses(prediction, ManualLabel).HD()]
                Dice[cnt,:] = [nucleiIx , smallFuncs.mDice(prediction, ManualLabel)]

                # confusionMatrix = metrics.confusionMatrix(predMV, ManualLabel)
                # Recall[cnt,:]    = [nucleiIx , confusionMatrix.Recall]
                # Precision[cnt,:] = [nucleiIx , confusionMatrix.Precision]
                    
        np.savetxt( self.dir_in + '/VSI_All.txt'       ,VSI , fmt='%1.1f %1.4f')
        np.savetxt( self.dir_in + '/HD_All.txt'        ,HD , fmt='%1.1f %1.4f')
        np.savetxt( self.dir_in + '/Dice_All.txt'      ,Dice , fmt='%1.1f %1.4f')
        # np.savetxt( self.dir_in + '/Recall_All.txt'    ,Recall , fmt='%1.1f %1.4f')
        # np.savetxt( self.dir_in + '/Precision_All.txt' ,Precision , fmt='%1.1f %1.4f')  
        
                  
    def loop_All_subjects(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'vimp' in s]:
            print(subj , '\n')
            temp = measure_Metrics_cls(dir_in= self.dir_in + '/' + subj , dir_label= self.dir_label + '/' + subj).measure_metrics()


UI = UserEntry()
# UI.dir_in  = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset2_with_Manual_Labels/full_Image/freesurfer/step2_freesurfer/Done/step2_resliced'
# UI.dir_label = '/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/Dataset2_with_Manual_Labels/full_Image/freesurfer/ManualLabels2_uncropped'
# UI.mode    = 'all'

if UI.mode == 'all':      
    measure_Metrics_cls(dir_in = UI.dir_in , dir_label = UI.dir_label).loop_All_subjects()
else:
    measure_Metrics_cls(dir_in = UI.dir_in , dir_label = UI.dir_label).measure_metrics()