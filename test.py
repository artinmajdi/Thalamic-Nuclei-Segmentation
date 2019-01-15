import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
import matplotlib.pyplot as plt

Dir = '/array/ssd/msmajdi/experiments/keras/exp1_tmp/models/subExp1_Loss_Dice/6-VLP/'

a  = open(Dir + 'hist_history.pkl' , 'rb')
b = pickle.load(a)
print(b)