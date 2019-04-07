import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import otherFuncs.smallFuncs as smallFuncs
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import pickle
import xlsxwriter
from tqdm import tqdm
import shutil

# UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
params = paramFunc.Run(UserInfo.__dict__, terminal=True)

def init_Params():
    _, NucleiIndexes , _ = smallFuncs.NucleiSelection(ind=1)
    NucleiIndexes = tuple(NucleiIndexes) + tuple([1.1,1.2,1.3])

    NumColumns = 27
    n_epochsMax = 300
    AllExp_Address = params.WhichExperiment.address

    return NucleiIndexes , NumColumns , n_epochsMax , AllExp_Address
NucleiIndexes , NumColumns , n_epochsMax , AllExp_Address = init_Params()

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
            # self.nucleus, _ , _ = smallFuncs.NucleiSelection(IxNu)
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

class Info_Search():    
    def __init__(self, General_Address=AllExp_Address , Experiment_Name = ''):
       
        class All_Experiments:
            address = General_Address
            List = [s for s in os.listdir(General_Address) if 'exp' in s] if General_Address else []
        self.All_Experiments = All_Experiments()
                 
        class Experiment:
            name                = Experiment_Name
            address             = self.All_Experiments.address + '/' + Experiment_Name
            List_subExperiments = ''
            TagsList             = []
        self.Experiment = Experiment()                                        
        
        def func_List_subExperiments(self, mode):
            
            class SD:
                def __init__(self, mode = False , name='' , sdx='' , TgC=0 , address=''):
                    self.mode = mode                        
                    if self.mode: 
                        self.tagList = np.append( ['Tag' + str(TgC) + '_' + sdx], name.split('_') ) 
                        # self.address = address
                        self.subject_List = [a for a in os.listdir(address) if 'vimp' in a]
                        self.subject_List.sort()
                        self.name = sdx
            class subExp():
                def __init__(self, name , address , TgC):
                    self.name = name  
                    # self.address = address + '/' + self.name

                    sdx = os.listdir(address + '/' + self.name)
                    sd0 = SD(True, name , 'sd0',TgC , address + '/' + name+'/sd0') if 'sd0' in sdx else SD() 
                    sd1 = SD(True, name , 'sd1',TgC , address + '/' + name+'/sd1') if 'sd1' in sdx else SD() 
                    sd2 = SD(True, name , 'sd2',TgC , address + '/' + name+'/sd2') if 'sd2' in sdx else SD() 

                    self.multiPlanar = [sd0 , sd1 , sd2]  
            
            List_subExps = [a for a in os.listdir(self.Experiment.address + '/' + mode) if ('subExp' in a) or ('sE' in a)]   

            self.Experiment.List_subExperiments , self.Experiment.TagsList = [] , []
            for Ix, name in enumerate(List_subExps):
                self.Experiment.List_subExperiments.append(subExp(name , self.Experiment.address + '/' + mode , Ix))
                self.Experiment.TagsList.append( np.append(['Tag' + str(Ix)],  name.split('_')) )

            # np.append( ['Tag' + str(TgC) + '_' + sdx], name.split('_') 
            # return [  for Ix, name in enumerate(List_subExps) ]        
        
        if self.Experiment.name:
            func_List_subExperiments(self, 'results')
            # self.Experiment.List_subExperiments = func_List_subExperiments(self, 'results')

        def func_Nuclei_Names():
            Nuclei_Names = np.append( ['subjects'] , list(np.zeros(NumColumns-1))  )
            Nuclei_Names[3] = ''
            def nuclei_Index(nIx):
                if nIx in range(15): return nIx
                elif nIx == 1.1:     return 15
                elif nIx == 1.2:     return 16
                elif nIx == 1.3:     return 17

            for nIx in NucleiIndexes:
                Nuclei_Names[nuclei_Index(nIx)] = smallFuncs.NucleiIndex(index=nIx).name

            return Nuclei_Names
        self.Nuclei_Names = func_Nuclei_Names()

for expName in Info_Search().All_Experiments.List:

    Info = Info_Search(Experiment_Name=expName )

    mergingDiceValues(Info)
    savingHistory_AsExcel(Info)

os.system('bash /array/ssd/msmajdi/code/thalamus/keras/bashCodes/zip_Bash_Merg')


# shutil.make_archive(base_name='All',format='zip',root_dir='/array/ssd/msmajdi/experiments/keras/exp*/results/All_*',)