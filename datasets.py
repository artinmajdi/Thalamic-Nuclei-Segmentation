
from keras.datasets import fashion_mnist, mnist

def loadDataset(dataset):

    if 'fashion_mnist' in dataset:
        data = fashion_mnist.load_data()
        class Train:
            Data  = data[0][0]
            Label = data[0][1]

        class Test:
            Data  = data[1][0]
            Label = data[1][1]


    return Train, Test
