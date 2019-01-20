import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
import matplotlib.pyplot as plt
import pandas as pd


Dir = '/array/ssd/msmajdi/experiments/keras/exp1_tmp/models/subExp3_Loss_Dice/8-Pul/'

a  = open(Dir + 'hist_params.pkl' , 'rb')
bb = pickle.load(a)

Data = [ str(bb[key]) for key in bb.keys()]
# for key in ListKeys: Data.append(str(bb[key]))

pd.DataFrame(data=bb,columns=list(bb.keys())).to_csv(Dir + 'test.csv')
print('----')