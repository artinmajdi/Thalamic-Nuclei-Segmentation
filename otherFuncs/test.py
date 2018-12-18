import numpy as np
from random import shuffle

a = np.random.random(10)
print(a)
a = np.array(range(10))
a
shuffle(a)
print(a)
a = np.random.random((10,20))
a = np.expand_dims(np.expand_dims(a,axis=0),axis=3)


np.concatenate((a,1-a),axis=3).shape
