import numpy as np
import random
from itertools import permutations

L = 4
AugNum = 5
if AugNum > L:
    AugNum = L-1

b = list(permutations(range(L),AugNum))

ind = 1
b = np.delete(C,C.index(ind))
b
C = [1,2,3,4,5]
random.shuffle(C)
C
