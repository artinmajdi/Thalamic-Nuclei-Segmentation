# #%%
import keras
# import numpy as np
# from keras.models import Sequential
# from keras.datasets import mnist , fashion_mnist
import os
import numpy as np
input1 = keras.layers.Input(shape=(10,30))
x1 = keras.layers.Dense(12, activation='relu')(input1)
input2 = keras.layers.Input(shape=(10,30))
x2 = keras.layers.Dense(12, activation='relu')(input2)

added = keras.layers.concatenate([x1,x2])
added2 = keras.backend.stack(added)
print(x1)
print(added)
print(added2)

class A():x=1

a = A()
a
