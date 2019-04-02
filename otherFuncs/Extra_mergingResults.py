import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import otherFuncs.smallFuncs as smallFuncs
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import pickle

UserInfoB = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
params = paramFunc.Run(UserInfoB)

def init_Params():
    _, NucleiIndexes , _ = smallFuncs.NucleiSelection(ind=1)
    NucleiIndexes = tuple(NucleiIndexes) + tuple([1.1,1.2,1.3])

    NumColumns = 27
    n_epochsMax = 300
    AllExp_Address = params.WhichExperiment.address

    return NucleiIndexes , NumColumns , n_epochsMax , AllExp_Address
NucleiIndexes , NumColumns , n_epochsMax , AllExp_Address = init_Params()

def func_Nuclei_Names():
    names = np.append( ['subjects'] , list(np.zeros(NumColumns-1))  )
    names[3] = ''
    for nIx in NucleiIndexes:
        if nIx in range(16): ind = nIx
        elif nIx == 1.1:     ind = 15
        elif nIx == 1.2:     ind = 16
        elif nIx == 1.3:     ind = 17

        names[ind], _ , _ = smallFuncs.NucleiSelection(nIx)

    return names

class savingHistory_AsExcel:

    def __init__(self, Info):

        self.Info = Info
        def Load_subDir_History(self):
                      
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

                columnsList = np.append([ 'Epochs', self.TagLst[0] ] ,  self.keys )
                self.FullNamesLA = np.append(self.FullNamesLA , columnsList)

                if self.ind == 0:
                    self.nucleusInfo[:self.N_Eps ,0] = np.array(range( self.N_Eps ))
                    self.AllNucleusInfo = self.nucleusInfo
                else:
                    self.AllNucleusInfo = np.concatenate((self.AllNucleusInfo, self.nucleusInfo) , axis=1)     

        writer = pd.ExcelWriter(  Info.Address + '/results/All_LossAccForEpochs.xlsx', engine='xlsxwriter')

        pd.DataFrame(data=self.Info.TagsList).to_excel(writer, sheet_name='TagsList')
        for IxNu in NucleiIndexes:
            self.nucleus, _ , _ = smallFuncs.NucleiSelection(IxNu)
            print('Learning Curves: ', self.nucleus)

            self.AllNucleusInfo , self.FullNamesLA , self.ind = [] , [] , -1

            for ix_sE , self.subExperiment in enumerate(self.Info.List_subExperiments):
                self.TagLst = self.Info.TagsList[ix_sE]
                self.subDir = self.Info.Address + '/models/' + self.subExperiment + '/' + self.nucleus                  
                if os.path.isdir(self.subDir): Load_subDir_History(self)

            if len(self.AllNucleusInfo) != 0: pd.DataFrame(data=self.AllNucleusInfo, columns=self.FullNamesLA).to_excel(writer, sheet_name=self.nucleus)

        writer.close()

class mergingDiceValues:
    def __init__(self, Info):

        self.Info = Info
        self.writer = pd.ExcelWriter(self.Info.Address + '/results/All_Dice.xlsx', engine='xlsxwriter')
        def mergingDiceValues_ForOneSubExperiment(self):

            print('Dices: ', str(self.subIx) + '/' + str(len(self.Info.List_subExperiments)), self.subExperiment)

            self.df_IndSubj = pd.DataFrame()
            self.TgLst = self.Info.TagsList[self.subIx]

            def subject_List(self):
                subjectsList = [a for a in os.listdir(self.subExperiment_Address) if 'vimp' in a]
                subjectsList.sort()
                return subjectsList
            
            def func_Load_AllNuclei_Dices(self):
                
                Dice_Single = np.append(self.subject, list(np.zeros(NumColumns-1)))
                
                for ind, name in enumerate( self.Nuclei_Names ):
                    Dir_subject = self.subExperiment_Address + '/' + self.subject + '/Dice_' + name +'.txt'
                    if os.path.isfile(Dir_subject): 
                        Dice_Single[ind] = np.loadtxt(Dir_subject)[1].astype(np.float16)

                # self.Dice_Test.append(Dice_Single)
                return Dice_Single
                
            self.Nuclei_Names = func_Nuclei_Names()
            self.Ind_Data = np.array([  func_Load_AllNuclei_Dices(self)  for self.subject in subject_List(self)  ]) 

            self.df_IndSubj[self.Nuclei_Names[0]] = self.Ind_Data[:,0]
            for nIx, nucleus in enumerate(self.Nuclei_Names[1:]): 
                self.df_IndSubj[nucleus] = self.Ind_Data[:,nIx+1].astype(np.float16)
            
            self.df_AD[self.TgLst[0]] = np.median(self.Ind_Data[:,1:].astype(np.float16),axis=0)[:17] 
            self.df_IndSubj.to_excel(  self.writer, sheet_name=self.TgLst[0] )         

        def save_TagList_AllDice(self):
            pd.DataFrame(data=self.Info.TagsList).to_excel(self.writer, sheet_name='TagsList')

            self.df_AD = pd.DataFrame()
            self.df_AD['Nuclei'] = func_Nuclei_Names()[1:18]
            self.df_AD.to_excel(self.writer, sheet_name='AllDices')        
        save_TagList_AllDice(self)

        for self.subIx, self.subExperiment in enumerate(self.Info.List_subExperiments):
            self.subExperiment_Address = self.Info.Address + '/results/' + self.subExperiment   
            try:         
                if os.path.isdir(self.subExperiment_Address): mergingDiceValues_ForOneSubExperiment(self)            
            except:
                print('failed' ,self.subExperiment )            
        self.df_AD.to_excel(self.writer, sheet_name='AllDices')
        self.writer.close()

class Info_Search():
    
    def __init__(self, All_Experiments_Address=AllExp_Address , Experiment_Name = ''):
                   
        def func_List_subExperiments(self):
            List_subExperiments_Results = [a for a in os.listdir(self.Address + '/results') if ('subExp' in a) or ('sE' in a)]
            List_subExperiments_Models = [a for a in os.listdir(self.Address + '/models') if ('subExp' in a) or ('sE' in a)]
            self.List_subExperiments = list(set(List_subExperiments_Results).union(List_subExperiments_Models))
            self.List_subExperiments.sort()

            self.TagsList = [np.append( ['Tag' + str(ixSE)], subEx.split('_') ) for ixSE, subEx in enumerate(self.List_subExperiments) ]
            
        self.All_Experiments_Address = All_Experiments_Address
        if All_Experiments_Address: self.All_Experiments_List = [s for s in os.listdir(All_Experiments_Address) if 'exp' in s]
    
        if Experiment_Name:
            self.subExperiment_Address = ''
            self.Experiment_Name = Experiment_Name  
            self.Address = self.All_Experiments_Address + '/' + self.Experiment_Name
            func_List_subExperiments(self)

for expName in Info_Search().All_Experiments_List:

    Info = Info_Search(Experiment_Name=expName )

    mergingDiceValues(Info)
    savingHistory_AsExcel(Info)

os.system('bash /array/ssd/msmajdi/code/thalamus/keras/bashCodes/zip_Bash_Merg')
