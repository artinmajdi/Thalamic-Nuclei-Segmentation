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
# params = paramFunc.Run(UserInfo.__dict__, terminal=True)

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

class merging_Dice_Values:
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

class merging_VSI_Values:
    def __init__(self, Info):
        self.Info   = Info
        self.writer = pd.ExcelWriter(Info.Experiment.address + '/results/All_VSI.xlsx', engine='xlsxwriter')
        
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
                                    
                ET    = insertNuclei_Excel(self, 'AllVSIs_ET')
                Main  = insertNuclei_Excel(self, 'AllVSIs_Main')
                CSFn1 = insertNuclei_Excel(self, 'AllVSIs_CSFn1')
                CSFn2 = insertNuclei_Excel(self, 'AllVSIs_CSFn2')
                SRI   = insertNuclei_Excel(self, 'AllVSIs_SRI')

            self.All_Subjs_Ns = All_Subjs_Ns()
        save_TagList(self)

        def func_Load_Subexperiment(self):
            
            # if self.plane.Flag:

            self.subExperiment.address = self.Info.Experiment.address + '/results/' + self.subExperiment.name +'/'+ self.plane.name
            
            def func_1subject_VSIs(self):                    
                                                                     
                def func_Search_Over_Single_Class(VSI_Single):
                    for name in All_Nuclei_Names:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + '/VSI_' + name +'.txt'
                        if os.path.isfile(Dir_subject): VSI_Single[name] = math.ceil( np.loadtxt(Dir_subject)[1] *1e3)/1e3
                    return VSI_Single
                
                def func_Search_Over_Multi_Class(VSI_Single):

                    for VSITag in ['/VSI_All' , '/VSI_All_Groups' , '/VSI_All_Medial' , '/VSI_All_lateral' , '/VSI_All_posterior']:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + VSITag + '.txt'
                        if os.path.isfile(Dir_subject): 
                            A = np.loadtxt(Dir_subject)

                            if not isinstance(A[0],np.ndarray): 
                                VSI_Single[smallFuncs.Nuclei_Class(index=A[0], method = 'HCascade').name] = math.ceil(  A[1]*1e3 )/1e3
                            else:
                                for id, nIx in enumerate(A[:,0]):
                                    VSI_Single[smallFuncs.Nuclei_Class(index=nIx, method = 'HCascade').name] = math.ceil(  A[id,1]*1e3 )/1e3

                    return VSI_Single

                VSI_Single = {'subject':self.subject} 
                VSI_Single = func_Search_Over_Single_Class(VSI_Single)   
                VSI_Single = func_Search_Over_Multi_Class(VSI_Single)

                return VSI_Single                                    
            sE_VSIs = np.array([  func_1subject_VSIs(self)  for self.subject in self.plane.subject_List  ])

            if len(sE_VSIs) > 0:
                def save_VSIs_subExp_In_ExcelFormat(self , sE_VSIs):
                    pd_sE = pd.DataFrame()

                    pd_sE['subject'] = [s['subject'] for s in sE_VSIs]
                    for nucleus in All_Nuclei_Names:

                        # try:
                        A = np.nan*np.ones(len(sE_VSIs))
                        for ix, s in enumerate(sE_VSIs):
                            if nucleus in s: A[ix] = s[nucleus]
                        
                        pd_sE[nucleus] = A

                        #     if nucleus in sE_VSIs[0]: pd_sE[nucleus] = [s[nucleus] for s in sE_VSIs] # .astype(np.float16)
                        # except Exception as e:
                            # print(e)

                    pd_sE.to_excel(  self.writer, sheet_name=self.plane.tagList[0] )    
                save_VSIs_subExp_In_ExcelFormat(self , sE_VSIs)
                
                def divideSubjects_BasedOnModality(self, sE_VSIs):
                    class subjectVSI:
                        ET   = []
                        Main = []
                        CSFn1 = []
                        CSFn2 = []
                        SRI  = []

                    # for sIx , subject in enumerate(sE_VSIs[:,0]):
                    for s in sE_VSIs:
                        if 'ET' in s['subject']:     subjectVSI.ET.append(s)
                        elif 'CSFn1' in s['subject']: subjectVSI.CSFn1.append(s)
                        elif 'CSFn2' in s['subject']: subjectVSI.CSFn2.append(s)
                        elif 'SRI'  in s['subject']: subjectVSI.SRI.append(s)
                        else:                        subjectVSI.Main.append(s)

                    def func_Average(subjectVSIList):

                        Average_VSIs = np.nan*np.ones(len(All_Nuclei_Names))
                        for ix, nucleus in enumerate(All_Nuclei_Names):
                            A = np.nan*np.ones(len(subjectVSIList))
                            for ct, s in enumerate(subjectVSIList):
                                if nucleus in s: A[ct] = s[nucleus]

                            Average_VSIs[ix] = np.round(1e3*np.nanmean(A, axis=0))/1e3
                                                        
                        return Average_VSIs

                    tag = self.plane.direction +'-' + self.plane.tagIndex    
                    # for dataset in ['ET' , 'Main' , '1' , 'SRI']:
                    #     A = subjectVSI.__getattribute__(dataset)
                    #     if len(A) > 0  : self.All_Subjs_Ns.__getattribute__(dataset).pd[tag] = func_Average(A)                   
                    #     # self.All_Subjs_Ns.__setattribute__(dataset) = A

                    if len(subjectVSI.ET) > 0   : self.All_Subjs_Ns.ET.pd[  tag]  = func_Average(subjectVSI.ET)   
                    if len(subjectVSI.Main) > 0 : self.All_Subjs_Ns.Main.pd[tag]  = func_Average(subjectVSI.Main) 
                    if len(subjectVSI.CSFn1) > 0: self.All_Subjs_Ns.CSFn1.pd[tag] = func_Average(subjectVSI.CSFn1) 
                    if len(subjectVSI.CSFn2) > 0: self.All_Subjs_Ns.CSFn2.pd[tag] = func_Average(subjectVSI.CSFn2) 
                    if len(subjectVSI.SRI) > 0  : self.All_Subjs_Ns.SRI.pd[ tag]  = func_Average(subjectVSI.SRI)  
                divideSubjects_BasedOnModality(self, sE_VSIs)

        def loopOver_Subexperiments(self):
            class smallActions():                                                                                              
                def add_space(self):
                    self.All_Subjs_Ns.ET.pd[  self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.Main.pd[self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn1.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn2.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.SRI.pd[ self.subExperiment.Tag[0]]  = np.nan*np.ones(17)

                def save_All_VSIs(PD):
                    PD.ET.pd.to_excel( self.writer , sheet_name=PD.ET.sheet_name  )
                    PD.Main.pd.to_excel(self.writer, sheet_name=PD.Main.sheet_name)
                    PD.CSFn1.pd.to_excel(self.writer, sheet_name=PD.CSFn1.sheet_name)
                    PD.CSFn2.pd.to_excel(self.writer, sheet_name=PD.CSFn2.sheet_name)
                    PD.SRI.pd.to_excel( self.writer, sheet_name=PD.SRI.sheet_name )

            A = zip(self.Info.Experiment.List_subExperiments , self.Info.Experiment.TagsList)
            for self.subExperiment, tag in tqdm( A , desc='VSIs:'):
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

            smallActions.save_All_VSIs(self.All_Subjs_Ns)

            self.writer.close()
        loopOver_Subexperiments(self)

class merging_HD_Values:
    def __init__(self, Info):
        self.Info   = Info
        self.writer = pd.ExcelWriter(Info.Experiment.address + '/results/All_HD.xlsx', engine='xlsxwriter')
        
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
                                    
                ET    = insertNuclei_Excel(self, 'AllHDs_ET')
                Main  = insertNuclei_Excel(self, 'AllHDs_Main')
                CSFn1 = insertNuclei_Excel(self, 'AllHDs_CSFn1')
                CSFn2 = insertNuclei_Excel(self, 'AllHDs_CSFn2')
                SRI   = insertNuclei_Excel(self, 'AllHDs_SRI')

            self.All_Subjs_Ns = All_Subjs_Ns()
        save_TagList(self)

        def func_Load_Subexperiment(self):
            
            # if self.plane.Flag:

            self.subExperiment.address = self.Info.Experiment.address + '/results/' + self.subExperiment.name +'/'+ self.plane.name
            
            def func_1subject_HDs(self):                    
                                                                     
                def func_Search_Over_Single_Class(HD_Single):
                    for name in All_Nuclei_Names:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + '/HD_' + name +'.txt'
                        if os.path.isfile(Dir_subject): HD_Single[name] = math.ceil( np.loadtxt(Dir_subject)[1] *1e3)/1e3
                    return HD_Single
                
                def func_Search_Over_Multi_Class(HD_Single):

                    for HDTag in ['/HD_All' , '/HD_All_Groups' , '/HD_All_Medial' , '/HD_All_lateral' , '/HD_All_posterior']:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + HDTag + '.txt'
                        if os.path.isfile(Dir_subject): 
                            A = np.loadtxt(Dir_subject)

                            if not isinstance(A[0],np.ndarray): 
                                HD_Single[smallFuncs.Nuclei_Class(index=A[0], method = 'HCascade').name] = math.ceil(  A[1]*1e3 )/1e3
                            else:
                                for id, nIx in enumerate(A[:,0]):
                                    if not np.isnan(A[id,1]): HD_Single[smallFuncs.Nuclei_Class(index=nIx, method = 'HCascade').name] = math.ceil(  A[id,1]*1e3 )/1e3
                                    else: HD_Single[smallFuncs.Nuclei_Class(index=nIx, method = 'HCascade').name] = np.nan

                    return HD_Single

                HD_Single = {'subject':self.subject} 
                HD_Single = func_Search_Over_Single_Class(HD_Single)   
                HD_Single = func_Search_Over_Multi_Class(HD_Single)

                return HD_Single                                    
            sE_HDs = np.array([  func_1subject_HDs(self)  for self.subject in self.plane.subject_List  ])

            if len(sE_HDs) > 0:
                def save_HDs_subExp_In_ExcelFormat(self , sE_HDs):
                    pd_sE = pd.DataFrame()

                    pd_sE['subject'] = [s['subject'] for s in sE_HDs]
                    for nucleus in All_Nuclei_Names:

                        # try:
                        A = np.nan*np.ones(len(sE_HDs))
                        for ix, s in enumerate(sE_HDs):
                            if nucleus in s: A[ix] = s[nucleus]
                        
                        pd_sE[nucleus] = A

                        #     if nucleus in sE_HDs[0]: pd_sE[nucleus] = [s[nucleus] for s in sE_HDs] # .astype(np.float16)
                        # except Exception as e:
                            # print(e)

                    pd_sE.to_excel(  self.writer, sheet_name=self.plane.tagList[0] )    
                save_HDs_subExp_In_ExcelFormat(self , sE_HDs)
                
                def divideSubjects_BasedOnModality(self, sE_HDs):
                    class subjectHD:
                        ET   = []
                        Main = []
                        CSFn1 = []
                        CSFn2 = []
                        SRI  = []

                    # for sIx , subject in enumerate(sE_HDs[:,0]):
                    for s in sE_HDs:
                        if 'ET' in s['subject']:     subjectHD.ET.append(s)
                        elif 'CSFn1' in s['subject']: subjectHD.CSFn1.append(s)
                        elif 'CSFn2' in s['subject']: subjectHD.CSFn2.append(s)
                        elif 'SRI'  in s['subject']: subjectHD.SRI.append(s)
                        else:                        subjectHD.Main.append(s)

                    def func_Average(subjectHDList):

                        Average_HDs = np.nan*np.ones(len(All_Nuclei_Names))
                        for ix, nucleus in enumerate(All_Nuclei_Names):
                            A = np.nan*np.ones(len(subjectHDList))
                            for ct, s in enumerate(subjectHDList):
                                if nucleus in s: A[ct] = s[nucleus]

                            Average_HDs[ix] = np.round(1e3*np.nanmean(A, axis=0))/1e3
                                                        
                        return Average_HDs

                    tag = self.plane.direction +'-' + self.plane.tagIndex    
                    # for dataset in ['ET' , 'Main' , '1' , 'SRI']:
                    #     A = subjectHD.__getattribute__(dataset)
                    #     if len(A) > 0  : self.All_Subjs_Ns.__getattribute__(dataset).pd[tag] = func_Average(A)                   
                    #     # self.All_Subjs_Ns.__setattribute__(dataset) = A

                    if len(subjectHD.ET) > 0   : self.All_Subjs_Ns.ET.pd[  tag]  = func_Average(subjectHD.ET)   
                    if len(subjectHD.Main) > 0 : self.All_Subjs_Ns.Main.pd[tag]  = func_Average(subjectHD.Main) 
                    if len(subjectHD.CSFn1) > 0: self.All_Subjs_Ns.CSFn1.pd[tag] = func_Average(subjectHD.CSFn1) 
                    if len(subjectHD.CSFn2) > 0: self.All_Subjs_Ns.CSFn2.pd[tag] = func_Average(subjectHD.CSFn2) 
                    if len(subjectHD.SRI) > 0  : self.All_Subjs_Ns.SRI.pd[ tag]  = func_Average(subjectHD.SRI)  
                divideSubjects_BasedOnModality(self, sE_HDs)

        def loopOver_Subexperiments(self):
            class smallActions():                                                                                              
                def add_space(self):
                    self.All_Subjs_Ns.ET.pd[  self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.Main.pd[self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn1.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn2.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.SRI.pd[ self.subExperiment.Tag[0]]  = np.nan*np.ones(17)

                def save_All_HDs(PD):
                    PD.ET.pd.to_excel( self.writer , sheet_name=PD.ET.sheet_name  )
                    PD.Main.pd.to_excel(self.writer, sheet_name=PD.Main.sheet_name)
                    PD.CSFn1.pd.to_excel(self.writer, sheet_name=PD.CSFn1.sheet_name)
                    PD.CSFn2.pd.to_excel(self.writer, sheet_name=PD.CSFn2.sheet_name)
                    PD.SRI.pd.to_excel( self.writer, sheet_name=PD.SRI.sheet_name )

            A = zip(self.Info.Experiment.List_subExperiments , self.Info.Experiment.TagsList)
            for self.subExperiment, tag in tqdm( A , desc='HDs:'):
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

            smallActions.save_All_HDs(self.All_Subjs_Ns)

            self.writer.close()
        loopOver_Subexperiments(self)

class merging_Volumes_Values:
    def __init__(self, Info):
        self.Info   = Info
        self.writer = pd.ExcelWriter(Info.Experiment.address + '/results/All_Volumes.xlsx', engine='xlsxwriter')
        
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
                                    
                ET    = insertNuclei_Excel(self, 'AllVolumess_ET')
                Main  = insertNuclei_Excel(self, 'AllVolumess_Main')
                CSFn1 = insertNuclei_Excel(self, 'AllVolumess_CSFn1')
                CSFn2 = insertNuclei_Excel(self, 'AllVolumess_CSFn2')
                SRI   = insertNuclei_Excel(self, 'AllVolumess_SRI')

            self.All_Subjs_Ns = All_Subjs_Ns()
        save_TagList(self)

        def func_Load_Subexperiment(self):
            
            # if self.plane.Flag:

            self.subExperiment.address = self.Info.Experiment.address + '/results/' + self.subExperiment.name +'/'+ self.plane.name
            
            def func_1subject_Volumess(self):                    
                                                                     
                def func_Search_Over_Single_Class(Volumes_Single):
                    for name in All_Nuclei_Names:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + '/Volumes_' + name +'.txt'
                        if os.path.isfile(Dir_subject): Volumes_Single[name] = math.ceil( np.loadtxt(Dir_subject)[1] *1e3)/1e3
                    return Volumes_Single
                
                def func_Search_Over_Multi_Class(Volumes_Single):

                    for VolumesTag in ['/Volumes_All' , '/Volumes_All_Groups' , '/Volumes_All_Medial' , '/Volumes_All_lateral' , '/Volumes_All_posterior']:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + VolumesTag + '.txt'
                        if os.path.isfile(Dir_subject): 
                            A = np.loadtxt(Dir_subject)

                            if not isinstance(A[0],np.ndarray): 
                                Volumes_Single[smallFuncs.Nuclei_Class(index=A[0], method = 'HCascade').name] = math.ceil(  A[1]*1e3 )/1e3
                            else:
                                for id, nIx in enumerate(A[:,0]):
                                    if not np.isnan(A[id,1]): Volumes_Single[smallFuncs.Nuclei_Class(index=nIx, method = 'HCascade').name] = math.ceil(  A[id,1]*1e3 )/1e3
                                    else: Volumes_Single[smallFuncs.Nuclei_Class(index=nIx, method = 'HCascade').name] = np.nan

                    return Volumes_Single

                Volumes_Single = {'subject':self.subject} 
                Volumes_Single = func_Search_Over_Single_Class(Volumes_Single)   
                Volumes_Single = func_Search_Over_Multi_Class(Volumes_Single)

                return Volumes_Single                                    
            sE_Volumess = np.array([  func_1subject_Volumess(self)  for self.subject in self.plane.subject_List  ])

            if len(sE_Volumess) > 0:
                def save_Volumess_subExp_In_ExcelFormat(self , sE_Volumess):
                    pd_sE = pd.DataFrame()

                    pd_sE['subject'] = [s['subject'] for s in sE_Volumess]
                    for nucleus in All_Nuclei_Names:

                        # try:
                        A = np.nan*np.ones(len(sE_Volumess))
                        for ix, s in enumerate(sE_Volumess):
                            if nucleus in s: A[ix] = s[nucleus]
                        
                        pd_sE[nucleus] = A

                        #     if nucleus in sE_Volumess[0]: pd_sE[nucleus] = [s[nucleus] for s in sE_Volumess] # .astype(np.float16)
                        # except Exception as e:
                            # print(e)

                    pd_sE.to_excel(  self.writer, sheet_name=self.plane.tagList[0] )    
                save_Volumess_subExp_In_ExcelFormat(self , sE_Volumess)
                
                def divideSubjects_BasedOnModality(self, sE_Volumess):
                    class subjectVolumes:
                        ET   = []
                        Main = []
                        CSFn1 = []
                        CSFn2 = []
                        SRI  = []

                    # for sIx , subject in enumerate(sE_Volumess[:,0]):
                    for s in sE_Volumess:
                        if 'ET' in s['subject']:     subjectVolumes.ET.append(s)
                        elif 'CSFn1' in s['subject']: subjectVolumes.CSFn1.append(s)
                        elif 'CSFn2' in s['subject']: subjectVolumes.CSFn2.append(s)
                        elif 'SRI'  in s['subject']: subjectVolumes.SRI.append(s)
                        else:                        subjectVolumes.Main.append(s)

                    def func_Average(subjectVolumesList):

                        Average_Volumess = np.nan*np.ones(len(All_Nuclei_Names))
                        for ix, nucleus in enumerate(All_Nuclei_Names):
                            A = np.nan*np.ones(len(subjectVolumesList))
                            for ct, s in enumerate(subjectVolumesList):
                                if nucleus in s: A[ct] = s[nucleus]

                            Average_Volumess[ix] = np.round(1e3*np.nanmean(A, axis=0))/1e3
                                                        
                        return Average_Volumess

                    tag = self.plane.direction +'-' + self.plane.tagIndex    
                    # for dataset in ['ET' , 'Main' , '1' , 'SRI']:
                    #     A = subjectVolumes.__getattribute__(dataset)
                    #     if len(A) > 0  : self.All_Subjs_Ns.__getattribute__(dataset).pd[tag] = func_Average(A)                   
                    #     # self.All_Subjs_Ns.__setattribute__(dataset) = A

                    if len(subjectVolumes.ET) > 0   : self.All_Subjs_Ns.ET.pd[  tag]  = func_Average(subjectVolumes.ET)   
                    if len(subjectVolumes.Main) > 0 : self.All_Subjs_Ns.Main.pd[tag]  = func_Average(subjectVolumes.Main) 
                    if len(subjectVolumes.CSFn1) > 0: self.All_Subjs_Ns.CSFn1.pd[tag] = func_Average(subjectVolumes.CSFn1) 
                    if len(subjectVolumes.CSFn2) > 0: self.All_Subjs_Ns.CSFn2.pd[tag] = func_Average(subjectVolumes.CSFn2) 
                    if len(subjectVolumes.SRI) > 0  : self.All_Subjs_Ns.SRI.pd[ tag]  = func_Average(subjectVolumes.SRI)  
                divideSubjects_BasedOnModality(self, sE_Volumess)

        def loopOver_Subexperiments(self):
            class smallActions():                                                                                              
                def add_space(self):
                    self.All_Subjs_Ns.ET.pd[  self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.Main.pd[self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn1.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn2.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.SRI.pd[ self.subExperiment.Tag[0]]  = np.nan*np.ones(17)

                def save_All_Volumess(PD):
                    PD.ET.pd.to_excel( self.writer , sheet_name=PD.ET.sheet_name  )
                    PD.Main.pd.to_excel(self.writer, sheet_name=PD.Main.sheet_name)
                    PD.CSFn1.pd.to_excel(self.writer, sheet_name=PD.CSFn1.sheet_name)
                    PD.CSFn2.pd.to_excel(self.writer, sheet_name=PD.CSFn2.sheet_name)
                    PD.SRI.pd.to_excel( self.writer, sheet_name=PD.SRI.sheet_name )

            A = zip(self.Info.Experiment.List_subExperiments , self.Info.Experiment.TagsList)
            for self.subExperiment, tag in tqdm( A , desc='Volumess:'):
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

            smallActions.save_All_Volumess(self.All_Subjs_Ns)

            self.writer.close()
        loopOver_Subexperiments(self)




UserInfoB = UserInfo.__dict__
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

params = paramFunc.Run(UserInfoB, terminal=True)

print(Experiment_Folder_Search(General_Address=params.WhichExperiment.address).All_Experiments.List)
for Experiment_Name in Experiment_Folder_Search(General_Address=params.WhichExperiment.address).All_Experiments.List[4:5]:

    print(Experiment_Name)
    Info = Experiment_Folder_Search(General_Address=params.WhichExperiment.address , Experiment_Name=Experiment_Name, mode='results')
    merging_Dice_Values(Info)
    merging_VSI_Values(Info)
    merging_HD_Values(Info)
    merging_Volumes_Values(Info)


    Info = Experiment_Folder_Search(General_Address=params.WhichExperiment.address , Experiment_Name=Experiment_Name, mode='models')    
    savingHistory_AsExcel(Info)

os.system('bash /array/ssd/msmajdi/code/thalamus/keras/bashCodes/zip_Bash_Merg')


# shutil.make_archive(base_name='All',format='zip',root_dir='/array/ssd/msmajdi/experiments/keras/exp*/results/All_*',)
