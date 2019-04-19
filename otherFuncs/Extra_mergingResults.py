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
import math
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
            pd.DataFrame(data=self.Info.Experiment.TagsList).to_excel(self.writer, sheet_name='TagsList')
            class All_Subjs_Ns:
                def insertNuclei_Excel(self, sheetName):
                    A = pd.DataFrame()
                    A['Nuclei'] = self.Info.Nuclei_Names[1:18]
                    A.to_excel(self.writer, sheet_name=sheetName)

                    class out:
                        pd = A
                        sheet_name = sheetName
                    return out()
                                    
                ET   = insertNuclei_Excel(self, 'AllDices_ET')
                Main = insertNuclei_Excel(self, 'AllDices_Main')
                CSFn = insertNuclei_Excel(self, 'AllDices_CSFn')

            self.All_Subjs_Ns = All_Subjs_Ns()
        save_TagList_AllDice(self)

        def func_Load_Subexperiment(self):
            
            if self.plane.mode:

                self.subExperiment.address = self.Info.Experiment.address + '/results/' + self.subExperiment.name +'/'+ self.plane.name
              
                def func_1subject_Dices(self):                    
                    Dice_Single = np.append(self.subject, list(np.zeros(NumColumns-1)))                    
                    for ind, name in enumerate( self.Info.Nuclei_Names ):
                        Dir_subject = self.subExperiment.address + '/' + self.subject + '/Dice_' + name +'.txt'
                        if os.path.isfile(Dir_subject): Dice_Single[ind] = math.ceil( np.loadtxt(Dir_subject)[1] *1e4)/1e4
                    return Dice_Single                                    
                sE_Dices = np.array([  func_1subject_Dices(self)  for self.subject in self.plane.subject_List  ])

                def save_Dices_subExp_In_ExcelFormat(self , sE_Dices):
                    pd_sE = pd.DataFrame()
                    for nIx, nucleus in enumerate(self.Info.Nuclei_Names): 
                        if nIx == 0 : pd_sE[nucleus] = sE_Dices[:,nIx]
                        else:         pd_sE[nucleus] = sE_Dices[:,nIx].astype(np.float16)
                    pd_sE.to_excel(  self.writer, sheet_name=self.plane.tagList[0] )    
                save_Dices_subExp_In_ExcelFormat(self , sE_Dices)
                
                def divideSubjects_BasedOnModality(self, sE_Dices):
                    class subjectDice:
                        ET   = []
                        Main = []
                        CSfn = []

                    for sIx , subject in enumerate(sE_Dices[:,0]):
                        # a = np.median(sE_Dices[subject,1:].astype(np.float),axis=0)[:17]
                        if len(subject) == 7:   subjectDice.ET.append(sIx)
                        elif 'CSFn' in subject: subjectDice.CSfn.append(sIx)
                        else:                   subjectDice.Main.append(sIx)

                    if len(subjectDice.ET) > 0:   self.All_Subjs_Ns.ET.pd[  self.plane.tagList[0]] = np.average(sE_Dices[subjectDice.ET, 1:18].astype(np.float) , axis=0)
                    if len(subjectDice.Main) > 0: self.All_Subjs_Ns.Main.pd[self.plane.tagList[0]] = np.median(sE_Dices[subjectDice.Main,1:18].astype(np.float) , axis=0)
                    if len(subjectDice.CSfn) > 0: self.All_Subjs_Ns.CSFn.pd[self.plane.tagList[0]] = np.median(sE_Dices[subjectDice.CSFn,1:18].astype(np.float) , axis=0)                        
                divideSubjects_BasedOnModality(self, sE_Dices)

        def loopOver_Subexperiments(self):
            class smallActions():                                                                                              
                def add_space(self):
                    self.All_Subjs_Ns.ET.pd[self.subExperiment.name]   = np.zeros(17)
                    self.All_Subjs_Ns.Main.pd[self.subExperiment.name] = np.zeros(17)
                    self.All_Subjs_Ns.CSFn.pd[self.subExperiment.name] = np.zeros(17)

                def save_All_Dices(PD):
                    PD.ET.pd.to_excel(self.writer , sheet_name=PD.ET.sheet_name)
                    PD.Main.pd.to_excel(self.writer, sheet_name=PD.Main.sheet_name)
                    PD.CSFn.pd.to_excel(self.writer, sheet_name=PD.CSFn.sheet_name)

            for self.subExperiment in tqdm( self.Info.Experiment.List_subExperiments , desc='Dices:'):

                smallActions.add_space(self)
                for self.plane in self.subExperiment.multiPlanar:
                    
                    try: func_Load_Subexperiment(self)
                    except: print('failed' ,self.subExperiment )                                            

            smallActions.save_All_Dices(self.All_Subjs_Ns)

            self.writer.close()
        loopOver_Subexperiments(self)


for Experiment_Name in Experiment_Folder_Search(General_Address=params.WhichExperiment.address).All_Experiments.List:

    Info = Experiment_Folder_Search(General_Address=params.WhichExperiment.address , Experiment_Name=Experiment_Name)

    mergingDiceValues(Info)
    # savingHistory_AsExcel(Info)

os.system('bash /array/ssd/msmajdi/code/thalamus/keras/bashCodes/zip_Bash_Merg')


# shutil.make_archive(base_name='All',format='zip',root_dir='/array/ssd/msmajdi/experiments/keras/exp*/results/All_*',)
