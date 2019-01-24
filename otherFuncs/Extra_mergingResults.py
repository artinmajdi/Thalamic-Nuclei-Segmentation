import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from otherFuncs import smallFuncs
from Parameters import UserInfo, paramFunc
import pickle 

def savingHistory_AsExcel(params):

        Dir = (params.directories.Train.Model).split('/subExp')[0]

        _, FullIndexes = smallFuncs.NucleiSelection(1)
        namesNulcei = smallFuncs.AllNucleiNames(FullIndexes)
        n_epochsMax = 300

        List_subExperiments = [a for a in os.listdir(Dir) if 'subExp' in a]
        writer = pd.ExcelWriter((params.directories.Test.Result).split('/subExp')[0] + '/All_LossAccDice.xlsx', engine='xlsxwriter')
        for nucleus in namesNulcei:
                # dir_save = smallFuncs.mkDir((params.directories.Test.Result).split('/subExp')[0] + '/Train_Output')
                AllNucleusInfo = []
                ind = -1
                for subExperiment in List_subExperiments:
                        subDir = Dir + '/' + subExperiment + '/' + nucleus
                        if os.path.exists(subDir):
                                ind = ind + 1
                                a  = open(subDir + '/hist_history.pkl' , 'rb')
                                history = pickle.load(a)
                                a.close()
                                keys = list(history.keys())
                                
                                nucleusInfo = np.zeros((n_epochsMax,len(keys)+2))
                                for ix, key in enumerate(keys):
                                        A = history[key]                                        
                                        nucleusInfo[1:len(A)+1,ix+2] = np.transpose(A)

                                if ind == 0:
                                        nucleusInfo[1:len(A)+1,0] = np.array(range(1,len(A)+1))
                                        AllNucleusInfo = nucleusInfo
                                        FullNamesLA = np.append(['Epochs', subExperiment],  keys )
                                else:
                                        AllNucleusInfo = np.concatenate((AllNucleusInfo, nucleusInfo) , axis=1)
                                        namesLA = np.append(['', subExperiment],  keys )
                                        FullNamesLA = np.append(FullNamesLA, namesLA)

                                # df = pd.DataFrame(data=nucleusInfo, columns=np.append(['Epochs', subExperiment],  keys ))
                                # df.to_csv(subDir + '/history.csv', index=False)
                                

                if len(AllNucleusInfo) != 0:
                        df = pd.DataFrame(data=AllNucleusInfo, columns=FullNamesLA)
                        # df.to_csv(Dir + '/history_AllSubExperiments_' + nucleus + '.csv', index=False)                        
                        df.to_excel(writer, sheet_name=nucleus)
        writer.close()

def mergingDiceValues(params):
        Dir = (params.directories.Test.Result).split('/subExp')[0]

        def mergingDiceValues_ForOneSubExperiment(Dir):
                subF = os.listdir(Dir)
                subF = [a for a in subF if 'vimp' in a]
                subF.sort()
                Dice_Test = []

                _, FullIndexes = smallFuncs.NucleiSelection(1)
                # names = np.append(['subjects'], smallFuncs.AllNucleiNames(FullIndexes))
                names = list(np.zeros(15))
                names[0] = 'subjects'
                for ind in FullIndexes:
                        if ind != 4567: 
                                names[ind], _ = smallFuncs.NucleiSelection(ind)

                for subject in subF:
                        Dir_subject = Dir + '/' + subject
                        a = os.listdir(Dir_subject)
                        a = [i for i in a if 'Dice_' in i]

                        Dice_Single = list(np.zeros(15))
                        Dice_Single[0] = subject
                        for n in a:
                                b = np.loadtxt(Dir_subject + '/' + n)
                                index = int(b[0])
                                if index != 4567: 
                                        Dice_Single[index] = b[1] 
                                else: 
                                        Dice_Single[3] = b[1] 
                                        
                        Dice_Test.append(Dice_Single)

                df = pd.DataFrame(data=Dice_Test, columns=names)
                df.to_csv(Dir + '/Dices.csv', index=False)

                return df

        writer = pd.ExcelWriter((params.directories.Test.Result).split('/subExp')[0] + '/All_HistoryData.xlsx', engine='xlsxwriter')
        List_subExperiments = [a for a in os.listdir(Dir) if 'subExp' in a]
        for subExperiment in List_subExperiments:
                df = mergingDiceValues_ForOneSubExperiment(Dir + '/' + subExperiment)
                if len(subExperiment) > 31: subExperiment = subExperiment[:31]
                df.to_excel(writer, sheet_name=subExperiment)

        writer.close()


UserInfo = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
params = paramFunc.Run(UserInfo)

mergingDiceValues(params)

savingHistory_AsExcel(params)

