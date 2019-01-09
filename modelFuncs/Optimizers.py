
from keras import optimizers


def OptimizerInfo(Optimizer_Index):
    switcher = {
        1: (optimizers.adam(), 'Adam'),
    }
    return switcher.get(Optimizer_Index, 'WARNING: Invalid Optimizer index')


