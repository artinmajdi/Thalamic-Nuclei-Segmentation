import os, sys
# sys.path.append(os.path.dirname(__file__))
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess

UserInfo = UserInfo.__dict__
# _ , UserInfo['simulation'].nucleus_Index,_ = smallFuncs.NucleiSelection(ind = 1)
UserInfoB = smallFuncs.terminalEntries(UserInfo)

class InitValues:
    Nuclei_Indexes = UserInfoB['simulation'].nucleus_Index.copy()
    slicingDim     = UserInfoB['simulation'].slicingDim.copy()

print('slicingDim' , InitValues.slicingDim , 'Nuclei_Indexes' , InitValues.Nuclei_Indexes , 'GPU:  ', UserInfoB['simulation'].GPU_Index)

def gpuSetting(params):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
    return K

def Run(UserInfoB):

    def HierarchicalStages(UserInfoB):

        def HCascade_Parents_Identifier(InitValues):
            b = smallFuncs.NucleiIndex(1,'HCascade')
            list_HC = []
            for ix in b.child:
                c = smallFuncs.NucleiIndex(ix,'HCascade')
                if c.child and bool(set(InitValues.Nuclei_Indexes) & set(c.child)): list_HC.append(ix)
            return list_HC

        def remove_Thalamus_From_List(InitValues):
            Nuclei_Indexes = InitValues.Nuclei_Indexes.copy()
            if 1 in Nuclei_Indexes: Nuclei_Indexes.remove(1)
            return Nuclei_Indexes

        print('************ stage 1 ************')
        #for UserInfoB['simulation'].nucleus_Index in [1]:
        #    Run_SingleNuclei(UserInfoB)

        print('************ stage 2 ************')               
        for UserInfoB['simulation'].nucleus_Index in HCascade_Parents_Identifier(InitValues):
            Run_SingleNuclei(UserInfoB)

        print('************ stage 3 ************')
        for UserInfoB['simulation'].nucleus_Index in remove_Thalamus_From_List(InitValues):
            Run_SingleNuclei(UserInfoB)

    def CacadeStages(UserInfoB):

        for UserInfoB['simulation'].nucleus_Index in InitValues.Nuclei_Indexes:
            Run_SingleNuclei(UserInfoB)

    def Run_SingleNuclei(UserInfoB):

        for sd in InitValues.slicingDim:

            UserInfoB['simulation'].slicingDim = [sd]     
                   
            UserInfoB['simulation'].epochs = 30 if UserInfoB['simulation'].nucleus_Index == 1 else 70
            params = paramFunc.Run(UserInfoB)

            print('---------------------------------------------------------------')
            print(' Nucleus:', UserInfoB['simulation'].nucleus_Index  , ' | GPU:', UserInfoB['simulation'].GPU_Index , ' | slicingDim',sd, \
                ' | Dropout', UserInfoB['DropoutValue'] , ' | Learning_Rate' , UserInfoB['simulation'].Learning_Rate, ' | num_Layers' , UserInfoB['simulation'].num_Layers,\
                ' | Multiply_By_Thalmaus',UserInfoB['simulation'].Multiply_By_Thalmaus )

            print('SubExperiment:', params.WhichExperiment.SubExperiment.name)
            print('---------------------------------------------------------------')
            Data, params = datasets.loadDataset(params)
            
            choosingModel.check_Run(params, Data)

    MethodType = params.WhichExperiment.HardParams.Model.Method.Type
    if   MethodType == 'HCascade':  HierarchicalStages(UserInfoB)
    elif MethodType == 'Cascade':   CacadeStages(UserInfoB)
    elif MethodType == 'singleRun': Run_SingleNuclei(UserInfoB)

params = paramFunc.Run(UserInfoB)
datasets.movingFromDatasetToExperiments(params)
applyPreprocess.main(params, 'experiment')
K = gpuSetting(params)



# # 1 & 2)
# UserInfoB['nucleus_Index'] = [1,2,8,9]
# UserInfoB['simulation'].Learning_Rate = 1e-3
# UserInfoB['simulation'].Multiply_By_Thalmaus = True
# for UserInfoB['DropoutValue'] in [0.3 , 0.4]:
#     try: Run(UserInfoB)
#     except: print('failed dropout' , UserInfoB['DropoutValue'])

# # 3 & 4)
# UserInfoB['nucleus_Index'] = [1,2,8,9]
# UserInfoB['DropoutValue'] = 0.3
# UserInfoB['simulation'].Multiply_By_Thalmaus = True
# for UserInfoB['simulation'].Learning_Rate in [1e-2 , 1e-3]:
#     try: Run(UserInfoB)
#     except: print('failed Learning_Rate' , UserInfoB['simulation'].Learning_Rate)

# # 5)
# UserInfoB['nucleus_Index'] = [1,2,8,9]
# UserInfoB['DropoutValue'] = 0.3
# UserInfoB['simulation'].Multiply_By_Thalmaus = True
# for UserInfoB['simulation'].Learning_Rate in [1e-4]:
#     try: Run(UserInfoB)
#     except: print('failed Learning_Rate' , UserInfoB['simulation'].Learning_Rate)

# # 6)
# UserInfoB['nucleus_Index'] = [1,2,8,9]
# UserInfoB['simulation'].Multiply_By_Thalmaus = False
# UserInfoB['DropoutValue'] = 0.3
# UserInfoB['simulation'].Learning_Rate = 1e-3
# try: Run(UserInfoB)
# except: print('failed Multiply_By_Thalmaus')

# # 7)
# UserInfoB['nucleus_Index'] = [1,2,8]
# UserInfoB['simulation'].Multiply_By_Thalmaus = True
# UserInfoB['DropoutValue'] = 0.3
# UserInfoB['simulation'].Learning_Rate = 1e-3
# UserInfoB['simulation'].Initialize_From_3T = True
# try: Run(UserInfoB)
# except: print('failed Initialize_From_3T')

# # 8)
# UserInfoB['nucleus_Index'] = [4,5,6,7,9,10,11,12,13,14]
# UserInfoB['ReadTrain'].ET   = False
# UserInfoB['ReadTrain'].Main = False
# UserInfoB['ReadTrain'].SRI  = True
# UserInfoB['SubExperiment'].Index = 8
# try: Run(UserInfoB)
# except: print('failed 3T')

# 9) HCascade
#UserInfoB['ReadTrain'].ET   = False
#UserInfoB['ReadTrain'].Main = False
#UserInfoB['ReadTrain'].SRI  = True
#UserInfoB['SubExperiment'].Index = 8
#try: 
Run(UserInfoB)
#except: print('failed 3T')

# # Need to fix ET cases first)
# UserInfoB['ReadTrain'].ET = True
# UserInfoB['DropoutValue'] = 0.3
# UserInfoB['simulation'].Learning_Rate = 1e-3
# try: Run(UserInfoB)
# except: print('failed wET')
# Learning_Rate

K.clear_session()
