
import os, sys
# __file__ = '/array/ssd/msmajdi/code/thalamus/keras/'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from otherFuncs import smallFuncs
from Parameters import UserInfo, paramFunc

def merginResults(Dir):
        subF = os.listdir(Dir)
        subF = [a for a in subF if 'vimp' in a]
        Dice_Test = []

        _, FullIndexes = smallFuncs.NucleiSelection(1)
        names = np.append(['subjects'], smallFuncs.AllNucleiNames(FullIndexes))

        for subject in subF:
                Dir_subject = Dir + '/' + subject
                a = os.listdir(Dir_subject)
                a = [i for i in a if 'Dice_' in i]

                Dice_Single = list(np.zeros(len(FullIndexes)+1))
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

        return Dice_Test

# params = paramFunc.Run(UserInfo.__dict__)
Dir = params.directories.Test.Result


Dir = '/media/data1/artin/a/subExp1_Loss_Dice'

merginResults(Dir)
