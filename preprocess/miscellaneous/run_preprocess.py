import os, sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from otherFuncs import smallFuncs
from modelFuncs import choosingModel
from Parameters import UserInfo, paramFunc
from preprocess import applyPreprocess

params = paramFunc.Run(UserInfo.__dict__, terminal=True)

applyPreprocess.main(params)

