import os
import sys

sys.path.append(os.path.dirname(__file__))
from Parameters import UserInfo
from otherFuncs.smallFuncs import terminalEntries
from full_multi_planar_framework import simulate

if __name__ == '__main__':
    UserEntry = terminalEntries(UserInfo.__dict__)
    UserEntry['simulation'] = UserEntry['simulation']()
    simulate(UserEntry)
