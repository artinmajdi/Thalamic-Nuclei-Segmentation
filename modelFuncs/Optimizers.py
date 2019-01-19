
def OptimizerInfo(Optimizer_Index):

    from keras import optimizers
    
    switcher = {
        1: (optimizers.adam(), 'Adam'),
    }
    return switcher.get(Optimizer_Index, 'WARNING: Invalid Optimizer index')


