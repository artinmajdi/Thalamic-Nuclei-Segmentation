
def OptimizerInfo(Optimizer_Index,learning_rate):

    from keras import optimizers
    
    switcher = {
        1: (optimizers.adam(lr=learning_rate), 'Adam'),
        # 2: (optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) , 'Adam'),
    }
    return switcher.get(Optimizer_Index, 'WARNING: Invalid Optimizer index')


