#%%
import keras
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist , fashion_mnist
import matplotlib.pyplot as plt
#%%
print(np.random.random(10))
print('finish')
data = mnist.load_data()


#%%
mode = Sequential()
plt.imshow(data[0][0][10] , cmap='gray')



#%%
data[0][0].max()


#%%
