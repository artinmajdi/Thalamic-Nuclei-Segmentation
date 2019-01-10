import os, sys
# __file__ = '/array/ssd/msmajdi/code/thalamus/keras/'  #! only if I'm using Hydrogen Atom
sys.path.append(os.path.dirname(__file__))
import main, experiment_Design

show = False



params = main.__init__()

AllparamsList = experiment_Design.main(params)

# Data, params = main.check_Dataset(params,'experiment')

# pred = main.check_Run(params, Data)

# if show: main.check_show(Data, pred)

