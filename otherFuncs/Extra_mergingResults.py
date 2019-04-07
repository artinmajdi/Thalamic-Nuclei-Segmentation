import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from otherFuncs.smallFuncs import Experiment_Folder_Search
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import pickle
import xlsxwriter
from tqdm import tqdm
import shutil

params = paramFunc.Run(UserInfo.__dict__, terminal=True)

NumColumns , n_epochsMax = 19 , 300

class savingHistory_AsExcel:
    def __init__(self, Info):

        self.Info = Info
        def Load_subDir_History(self):
            self.subDir = self.Info.Experiment.address + '/models/' + self.subExperiment.name + '/' + self.nucleus + '/' + self.plane.name
            if os.path.isfile( self.subDir + '/hist_history.pkl' ):
                self.ind += 1                
                def load_NucleusHisotry(self):
                    with  open(self.subDir + '/hist_history.pkl' , 'rb') as a:  
                        history = pickle.load(a)

                    self.keys = list(history.keys())
                    self.nucleusInfo = np.zeros((n_epochsMax,len(self.keys)+2))
                    for ix, key in enumerate(self.keys):
                        self.N_Eps = len(history[key])
                        self.nucleusInfo[:self.N_Eps,ix+2] = history[key]
                load_NucleusHisotry(self)

                columnsList = np.append([ 'Epochs', self.plane.tagList[0] ] ,  self.keys )
                self.FullNamesLA = np.append(self.FullNamesLA , columnsList)

                if self.ind == 0:
                    self.nucleusInfo[:self.N_Eps ,0] = np.array(range( self.N_Eps ))
                    self.AllNucleusInfo = self.nucleusInfo
                else:
                    self.AllNucleusInfo = np.concatenate((self.AllNucleusInfo, self.nucleusInfo) , axis=1)     

        writer = pd.ExcelWriter(  Info.Experiment.address + '/results/All_LossAccForEpochs.xlsx', engine='xlsxwriter')
        
        pd.DataFrame(data=self.Info.Experiment.TagsList).to_excel(writer, sheet_name='TagsList')
        for self.nucleus in Info.Nuclei_Names[1:17]:
            print('Learning Curves: ', self.nucleus)

            self.AllNucleusInfo , self.FullNamesLA , self.ind = [] , [] , -1

            for self.subExperiment in self.Info.Experiment.List_subExperiments:
                for self.plane in self.subExperiment.multiPlanar:
                    if self.plane.mode: Load_subDir_History(self)

            if len(self.AllNucleusInfo) != 0: pd.DataFrame(data=self.AllNucleusInfo, columns=self.FullNamesLA).to_excel(writer, sheet_name=self.nucleus)

        writer.close()

class mergingDiceValues:
    def __init__(self, Info):
        self.Info   = Info
        self.writer = pd.ExcelWriter(self.Info.Experiment.address + '/results/All_Dice.xlsx', engine='xlsxwriter')
        def save_TagList_AllDice(self):
            # TODO it might be unnecessary
            pd.DataFrame(data=self.Info.Experiment.TagsList).to_excel(self.writer, sheet_name='TagsList')

            self.pd_AllNuclei_Dices = pd.DataFrame()
            self.pd_AllNuclei_Dices['Nuclei'] = self.Info.Nuclei_Names[1:18]
            self.pd_AllNuclei_Dices.to_excel(self.writer, sheet_name='AllDices')        
        save_TagList_AllDice(self)

        def func_Load_Subexperiment(self):
            
            if self.plane.mode:

                self.subExperiment.address = self.Info.Experiment.address + '/results/' + self.subExperiment.name +'/'+ self.plane.name
              
                def func_1subject_Dices(self):
                    
                    Dice_Single = np.append(self.subject, list(np.zeros(NumColumns-1)))
                    
                    for ind, name in enumerate( self.Info.Nuclei_Names ):
                        Dir_subject = self.subExperiment.address + '/' + self.subject + '/Dice_' + name +'.txt'
                        if os.path.isfile(Dir_subject): Dice_Single[ind] = np.loadtxt(Dir_subject)[1].astype(np.float16)
                    # self.Dice_Test.append(Dice_Single)
                    return Dice_Single                    
                # func_Nuclei_Names(self)
                sE_Dices = np.array([  func_1subject_Dices(self)  for self.subject in self.plane.subject_List  ]) 

                def save_Dices_subExp_In_ExcelFormat(self , sE_Dices):
                    pd_sE = pd.DataFrame()
                    for nIx, nucleus in enumerate(self.Info.Nuclei_Names): 
                        if nIx == 0 : pd_sE[nucleus] = sE_Dices[:,nIx]
                        else:         pd_sE[nucleus] = sE_Dices[:,nIx].astype(np.float16)
                    pd_sE.to_excel(  self.writer, sheet_name=self.plane.tagList[0] )    
                save_Dices_subExp_In_ExcelFormat(self , sE_Dices)

                self.pd_AllNuclei_Dices[self.plane.tagList[0]] = np.median(sE_Dices[:,1:].astype(np.float16),axis=0)[:17] 
                     
        def loopOver_Subexperiments(self):
            for self.subExperiment in tqdm( self.Info.Experiment.List_subExperiments , desc='Dices:'):
                for self.plane in self.subExperiment.multiPlanar:
                    
                    try: func_Load_Subexperiment(self)
                    except: print('failed' ,self.subExperiment )            
                
            self.pd_AllNuclei_Dices.to_excel(self.writer, sheet_name='AllDices')
            self.writer.close()
        loopOver_Subexperiments(self)


for Experiment_Name in Experiment_Folder_Search(General_Address=params.WhichExperiment.address).All_Experiments.List:

    Info = Experiment_Folder_Search(General_Address=params.WhichExperiment.address , Experiment_Name=Experiment_Name)

    mergingDiceValues(Info)
    savingHistory_AsExcel(Info)

os.system('bash /array/ssd/msmajdi/code/thalamus/keras/bashCodes/zip_Bash_Merg')


# shutil.make_archive(base_name='All',format='zip',root_dir='/array/ssd/msmajdi/experiments/keras/exp*/results/All_*',)