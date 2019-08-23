import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras/')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import otherFuncs.smallFuncs as smallFuncs
import preprocess.applyPreprocess as applyPreprocess
import Parameters.paramFunc as paramFunc
import Parameters.UserInfo as UserInfo

# UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)
UserInfoB = UserInfo.__dict__
UserInfoB['simulation'].TestOnly = True
UserInfoB['TypeExperiment'] = 11
UserInfoB['Model_Method'] = 'Cascade' 
UserInfoB['architectureType'] = 'ResFCN_ResUnet2_TL'
UserInfoB['lossFunction_Index'] = 4
# UserInfoB['Experiments'].Index = '6'
UserInfoB['copy_Thalamus'] = False
UserInfoB['simulation'].batch_size = 50
UserInfoB['simulation'].FirstLayer_FeatureMap_Num = 20    
UserInfoB['simulation'].FCN1_NLayers = 0
UserInfoB['simulation'].FCN2_NLayers = 0  
UserInfoB['simulation'].FCN_FeatureMaps = 0
UserInfoB['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]
UserInfoB['simulation'].TestOnly = True
UserInfoB['CrossVal'].index      = ['a']
UserInfoB['Experiments'].Index = '10'
UserInfoB['Experiments'].Tag = 'test_Manoj'

params = paramFunc.Run(UserInfoB, terminal=True)
params.preprocess.Mode = True

applyPreprocess.main(params, 'experiment')

