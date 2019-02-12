import numpy as np


a = ((-2,9),(5,8),(-5,2),(0,-9))

b = [list(i) for i in a]

b
ba = np.array(b)
ba[ba > 0] = 0
ba *= -1
ba
ba.shape
np.where(ba < 0)[0].shape

np.where(b[0] < 0)





a = np.zeros((20,20))

a = a[2:,:]
a = a[:,:-3]
a.shape





























print('----')
