import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from otherFuncs.smallFuncs import Experiment_Folder_Search
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import otherFuncs.smallFuncs as smallFuncs
import pickle
import xlsxwriter
from tqdm import tqdm
import math
# import shutil
params = paramFunc.Run(UserInfo.__dict__, terminal=True)

NumColumns , n_epochsMax = 19 , 300

All_Nuclei_Names = smallFuncs.Nuclei_Class(method = 'HCascade').All_Nuclei().Names
All_Nuclei_Indexes = smallFuncs.Nuclei_Class(method = 'HCascade').All_Nuclei().Indexes

class savingHistory_AsExcel:
    def __init__(self, Info):

        self.Info = Info
        def Load_subDir_History(self):
            self.subDir = self.Info.Experiment.address + '/models/' + self.subExperiment.name + '/' + self.nucleus + '/' + self.plane.name
            if os.path.isfile( self.subDir + '/hist_history.pkl' ):
                self.ind += 1                
                def load_NucleusHisotry(self):
                    with open(self.subDir + '/hist_history.pkl' , 'rb') as aa:
                        history = pickle.load(aa)
                        keys = list(history.keys())
                    
                    self.nucleusInfo = np.nan*np.ones((n_epochsMax,len(keys)+2))
                    for ix, key in enumerate(keys):
                        self.N_Eps = L = len(history[key])
                        data = np.array(history[key])

                        if   'Dice' in key: self.nucleusInfo[:L,ix+2] = np.round(1e3*data)/1e3
                        elif 'loss' in key: self.nucleusInfo[:L,ix+2] = np.round(1e5*data)/1e5
                        elif 'acc'  in key: self.nucleusInfo[:L,ix+2] = np.round(1e5*data)/1e5
                        elif 'lr'   in key: self.nucleusInfo[:L,ix+2] = np.round(1e5*data)/1e5
                        else:               self.nucleusInfo[:L,ix+2] = data

                    return keys

                keys = load_NucleusHisotry(self)

                columnsList = np.append([ 'Epochs', self.plane.tagList[0] ] ,  keys )
                self.FullNamesLA = np.append(self.FullNamesLA , columnsList)

                if self.ind == 0:
                    self.nucleusInfo[:self.N_Eps ,0] = np.array(range( self.N_Eps ))
                    self.AllNucleusInfo = self.nucleusInfo
                else:
                    self.AllNucleusInfo = np.concatenate((self.AllNucleusInfo, self.nucleusInfo) , axis=1)     

        def find_nuclei_models_list():
            nuclei_List = set()
            for self.subExperiment in self.Info.Experiment.List_subExperiments:
                address = Info.Experiment.address + '/models/' + self.subExperiment.name
                A = set(os.listdir(address))
                nuclei_List = nuclei_List.union(A)

            nuclei_List = list(nuclei_List)
            nuclei_List.sort()
            return nuclei_List

        writer = pd.ExcelWriter(  Info.Experiment.address + '/results/All_LossAccForEpochs.xlsx', engine='xlsxwriter')
        


        pd.DataFrame(data=self.Info.Experiment.TagsList).to_excel(writer, sheet_name='TagsList')
        for self.nucleus in find_nuclei_models_list():
            print('Learning Curves: ', self.nucleus)

            self.AllNucleusInfo , self.FullNamesLA , self.ind = [] , [] , -1

            for self.subExperiment in self.Info.Experiment.List_subExperiments:
                for self.plane in self.subExperiment.multiPlanar:
                    if self.plane.Flag: Load_subDir_History(self)

            if len(self.AllNucleusInfo) != 0: pd.DataFrame(data=self.AllNucleusInfo, columns=self.FullNamesLA).to_excel(writer, sheet_name=self.nucleus.replace('_ImClosed',''))

        writer.close()

class mergingDiceValues:
    def __init__(self, Info):
        self.Info   = Info
        self.writer = pd.ExcelWriter(Info.Experiment.address + '/results/All_Dice.xlsx', engine='xlsxwriter')
        
        def save_TagList(self):
            pd.DataFrame(data=self.Info.Experiment.TagsList).to_excel(self.writer, sheet_name='TagsList')
            class All_Subjs_Ns:
                def insertNuclei_Excel(self, sheetName):
                    A = pd.DataFrame()
                    A['Nuclei'] = All_Nuclei_Names # self.Info.Nuclei_Names[1:18]
                    A.to_excel(self.writer, sheet_name=sheetName)

                    class out:
                        pd = A
                        sheet_name = sheetName
                    return out()
                                    
                ET    = insertNuclei_Excel(self, 'AllDices_ET')
                Main  = insertNuclei_Excel(self, 'AllDices_Main')
                CSFn1 = insertNuclei_Excel(self, 'AllDices_CSFn1')
                CSFn2 = insertNuclei_Excel(self, 'AllDices_CSFn2')
                SRI   = insertNuclei_Excel(self, 'AllDices_SRI')

            self.All_Subjs_Ns = All_Subjs_Ns()
        save_TagList(self)

        def func_Load_Subexperiment(self):
            
            # if self.plane.Flag:

            self.subExperiment.address = self.Info.Experiment.address + '/results/' + self.subExperiment.name +'/'+ self.plane.name
            
            def func_1subject_Dices(self):                    
                                                                     
                def func_Search_Over_Single_Class(Dice_Single):
                    for name in All_Nuclei_Names:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + '/Dice_' + name +'.txt'
                        if os.path.isfile(Dir_subject): Dice_Single[name] = math.ceil( np.loadtxt(Dir_subject)[1] *1e3)/1e3
                    return Dice_Single
                
                def func_Search_Over_Multi_Class(Dice_Single):

                    for DiceTag in ['/Dice_All' , '/Dice_All_Groups' , '/Dice_All_Medial' , '/Dice_All_lateral' , '/Dice_All_posterior']:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + DiceTag + '.txt'
                        if os.path.isfile(Dir_subject): 
                            A = np.loadtxt(Dir_subject)

                            if not isinstance(A[0],np.ndarray): 
                                Dice_Single[smallFuncs.Nuclei_Class(index=A[0], method = 'HCascade').name] = math.ceil(  A[1]*1e3 )/1e3
                            else:
                                for id, nIx in enumerate(A[:,0]):
                                    Dice_Single[smallFuncs.Nuclei_Class(index=nIx, method = 'HCascade').name] = math.ceil(  A[id,1]*1e3 )/1e3

                    return Dice_Single

                Dice_Single = {'subject':self.subject} 
                Dice_Single = func_Search_Over_Single_Class(Dice_Single)   
                Dice_Single = func_Search_Over_Multi_Class(Dice_Single)

                return Dice_Single                                    
            sE_Dices = np.array([  func_1subject_Dices(self)  for self.subject in self.plane.subject_List  ])

            if len(sE_Dices) > 0:
                def save_Dices_subExp_In_ExcelFormat(self , sE_Dices):
                    pd_sE = pd.DataFrame()

                    pd_sE['subject'] = [s['subject'] for s in sE_Dices]
                    for nucleus in All_Nuclei_Names:

                        # try:
                        A = np.nan*np.ones(len(sE_Dices))
                        for ix, s in enumerate(sE_Dices):
                            if nucleus in s: A[ix] = s[nucleus]
                        
                        pd_sE[nucleus] = A

                        #     if nucleus in sE_Dices[0]: pd_sE[nucleus] = [s[nucleus] for s in sE_Dices] # .astype(np.float16)
                        # except Exception as e:
                            # print(e)

                    pd_sE.to_excel(  self.writer, sheet_name=self.plane.tagList[0] )    
                save_Dices_subExp_In_ExcelFormat(self , sE_Dices)
                
                def divideSubjects_BasedOnModality(self, sE_Dices):
                    class subjectDice:
                        ET   = []
                        Main = []
                        CSFn1 = []
                        CSFn2 = []
                        SRI  = []

                    # for sIx , subject in enumerate(sE_Dices[:,0]):
                    for s in sE_Dices:
                        if 'ET' in s['subject']:     subjectDice.ET.append(s)
                        elif 'CSFn1' in s['subject']: subjectDice.CSFn1.append(s)
                        elif 'CSFn2' in s['subject']: subjectDice.CSFn2.append(s)
                        elif 'SRI'  in s['subject']: subjectDice.SRI.append(s)
                        else:                        subjectDice.Main.append(s)

                    def func_Average(subjectDiceList):

                        Average_Dices = np.nan*np.ones(len(All_Nuclei_Names))
                        for ix, nucleus in enumerate(All_Nuclei_Names):
                            A = np.nan*np.ones(len(subjectDiceList))
                            for ct, s in enumerate(subjectDiceList):
                                if nucleus in s: A[ct] = s[nucleus]

                            Average_Dices[ix] = np.round(1e3*np.nanmean(A, axis=0))/1e3
                                                        
                        return Average_Dices

                    tag = self.plane.direction +'-' + self.plane.tagIndex    
                    # for dataset in ['ET' , 'Main' , '1' , 'SRI']:
                    #     A = subjectDice.__getattribute__(dataset)
                    #     if len(A) > 0  : self.All_Subjs_Ns.__getattribute__(dataset).pd[tag] = func_Average(A)                   
                    #     # self.All_Subjs_Ns.__setattribute__(dataset) = A

                    if len(subjectDice.ET) > 0   : self.All_Subjs_Ns.ET.pd[  tag]  = func_Average(subjectDice.ET)   
                    if len(subjectDice.Main) > 0 : self.All_Subjs_Ns.Main.pd[tag]  = func_Average(subjectDice.Main) 
                    if len(subjectDice.CSFn1) > 0: self.All_Subjs_Ns.CSFn1.pd[tag] = func_Average(subjectDice.CSFn1) 
                    if len(subjectDice.CSFn2) > 0: self.All_Subjs_Ns.CSFn2.pd[tag] = func_Average(subjectDice.CSFn2) 
                    if len(subjectDice.SRI) > 0  : self.All_Subjs_Ns.SRI.pd[ tag]  = func_Average(subjectDice.SRI)  
                divideSubjects_BasedOnModality(self, sE_Dices)

        def loopOver_Subexperiments(self):
            class smallActions():                                                                                              
                def add_space(self):
                    self.All_Subjs_Ns.ET.pd[  self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.Main.pd[self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn1.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn2.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.SRI.pd[ self.subExperiment.Tag[0]]  = np.nan*np.ones(17)

                def save_All_Dices(PD):
                    PD.ET.pd.to_excel( self.writer , sheet_name=PD.ET.sheet_name  )
                    PD.Main.pd.to_excel(self.writer, sheet_name=PD.Main.sheet_name)
                    PD.CSFn1.pd.to_excel(self.writer, sheet_name=PD.CSFn1.sheet_name)
                    PD.CSFn2.pd.to_excel(self.writer, sheet_name=PD.CSFn2.sheet_name)
                    PD.SRI.pd.to_excel( self.writer, sheet_name=PD.SRI.sheet_name )

            A = zip(self.Info.Experiment.List_subExperiments , self.Info.Experiment.TagsList)
            for self.subExperiment, tag in tqdm( A , desc='Dices:'):
                # try: 
                self.subExperiment.Tag = tag
                # print(self.subExperiment.name)
                smallActions.add_space(self)
                for self.plane in self.subExperiment.multiPlanar:
                    if self.plane.Flag:
                        # print(self.subExperiment.name , self.plane.name)
                        # try: 
                        func_Load_Subexperiment(self)
                        # except: print('failed' ,self.subExperiment )                                            
                # except Exception as e:
                #     print(e)

            smallActions.save_All_Dices(self.All_Subjs_Ns)

            self.writer.close()
        loopOver_Subexperiments(self)


print(Experiment_Folder_Search(General_Address=params.WhichExperiment.address).All_Experiments.List)
for Experiment_Name in Experiment_Folder_Search(General_Address=params.WhichExperiment.address).All_Experiments.List[-3:-1]:

    # print(Experiment_Name)
    Info = Experiment_Folder_Search(General_Address=params.WhichExperiment.address , Experiment_Name=Experiment_Name, mode='results')
    mergingDiceValues(Info)

    # Info = Experiment_Folder_Search(General_Address=params.WhichExperiment.address , Experiment_Name=Experiment_Name, mode='models')    
    # savingHistory_AsExcel(Info)

os.system('bash /array/ssd/msmajdi/code/thalamus/keras/bashCodes/zip_Bash_Merg')


# shutil.make_archive(base_name='All',format='zip',root_dir='/array/ssd/msmajdi/experiments/keras/exp*/results/All_*',)
