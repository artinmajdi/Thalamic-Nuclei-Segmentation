import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from Parameters import UserInfo
from otherFuncs.smallFuncs import terminalEntries
from full_multi_planar_framework import simulate

if __name__ == '__main__':
    UserEntry = terminalEntries(UserInfo.__dict__)
    simulate(UserEntry)
