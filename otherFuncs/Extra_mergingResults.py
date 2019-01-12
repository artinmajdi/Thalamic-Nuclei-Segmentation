
import os, sys
__file__ = '/array/ssd/msmajdi/code/thalamus/keras/'
sys.path.append(os.path.dirname(__file__))
import numpy as np
from Parameters import UserInfo, paramFunc

params = paramFunc.Run(UserInfo.__dict__)

Dir = params.directories.Test.Result
subF = os.listdir(Dir)

Dice_Test = (np.zeros((len(subF), 14)))
for ind, subject in enumerate(subF):
    Dir_subject = Dir + '/' + subject
    a = os.listdir(Dir_subject)
    a = [i for i in a if 'Dice_' in i]

    for n in a:
        b = np.loadtxt(Dir_subject + '/' + n)
        Dice_Test[ ind, int(b[0])] = b[1]




print('----------')
print(Dice_Test)

