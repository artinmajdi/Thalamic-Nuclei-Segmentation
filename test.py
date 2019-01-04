import os, sys
__file__ = '/array/ssd/msmajdi/code/thalamus/keras'  #! only if I'm using Hydrogen Atom
sys.path.append(__file__)
from astropy import table, units


a = table.Table([[1,5,6,7],[2,3,1,4]],names=('dataset','b'))

a

a.add_column('aaa')
a.add_column(table.column('aaa',names=('f'),units=units.strin))
