
from keras.datasets import fashion_mnist, mnist
import numpy as np
# from keras.backend import one_hot

def one_hot(a, num_classes):
  return np.eye(num_classes)[a]

def loadDataset(dataset):

    if 'fashion_mnist' in dataset:
        data = fashion_mnist.load_data()
        class Train:
            Data  = (np.expand_dims(data[0][0],axis=3)).astype('float32') / 255
            Label = one_hot(data[0][1],10)
            # Label = data[0][1]

        class Test:
            Data  = (np.expand_dims(data[1][0],axis=3)).astype('float32') / 255
            Label = one_hot(data[1][1],10)
            # Label = data[1][1]


    return Train, Test
