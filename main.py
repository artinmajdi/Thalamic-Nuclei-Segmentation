import os, sys
sys.path.append(os.path.dirname(__file__))
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import otherFuncs.smallFuncs as smallFuncs
import otherFuncs.datasets as datasets
import modelFuncs.choosingModel as choosingModel
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import preprocess.applyPreprocess as applyPreprocess
from shutil import copyfile , copytree
import tensorflow as tf
from keras import backend as K

class InitValues:
    def __init__(self, Nuclei_Indexes=1 , slicingDim=2):
        self.slicingDim     = slicingDim.copy()

        if Nuclei_Indexes == 'all':  
             _, self.Nuclei_Indexes,_ = smallFuncs.NucleiSelection(ind = 1)
        else:
            self.Nuclei_Indexes = Nuclei_Indexes.copy()
                
def gpuSetting(params):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
    # K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True )   ))
    return K

def Run(UserInfoB,InitValues):

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
        if 1 in InitValues.Nuclei_Indexes: 
            for UserInfoB['simulation'].nucleus_Index in [1]:
                Run_SingleNuclei(UserInfoB)

        print('************ stage 2 ************')               
        # for UserInfoB['simulation'].nucleus_Index in HCascade_Parents_Identifier(InitValues):
        #     Run_SingleNuclei(UserInfoB)

        print('************ stage 3 ************')
        for UserInfoB['simulation'].nucleus_Index in remove_Thalamus_From_List(InitValues):
            Run_SingleNuclei(UserInfoB)

    def CacadeStages(UserInfoB):

        for UserInfoB['simulation'].nucleus_Index in InitValues.Nuclei_Indexes:
            Run_SingleNuclei(UserInfoB)

    def Run_SingleNuclei(UserInfoB):

        for sd in InitValues.slicingDim:
            
            K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))

            UserInfoB['simulation'].slicingDim = [sd]                       
            params = paramFunc.Run(UserInfoB)

            print('---------------------------------------------------------------')
            print(' Nucleus:', UserInfoB['simulation'].nucleus_Index  , ' | GPU:', UserInfoB['simulation'].GPU_Index , ' | slicingDim',sd, \
                ' | Dropout', UserInfoB['DropoutValue'] , ' | Learning_Rate' , UserInfoB['simulation'].Learning_Rate, ' | num_Layers' , UserInfoB['simulation'].num_Layers,\
                ' | Multiply_By_Thalmaus',UserInfoB['simulation'].Multiply_By_Thalmaus )

            print('SubExperiment:', params.WhichExperiment.SubExperiment.name)
            print('---------------------------------------------------------------')
              

            # # try: 
            # if sd == 0 and UserInfoB['simulation'].nucleus_Index == 1: 
                
            #     os.system("mkdir %s"%())

            #     UserInfoC = UserInfoB.copy()  
                
            #     UserInfoC['simulation'].slicingDim = [2]
            #     paramsC = paramFunc.Run(UserInfoC)
            #     Data, paramsC = datasets.loadDataset(params) 

            #     UserInfoC['simulation'].TestOnly = True
            #     UserInfoC['simulation'].slicingDim = [0]
            #     paramsC = paramFunc.Run(UserInfoC)
            #     paramsC.directories.Train.Model          = paramsC.directories.Train.Model.replace('sd0','sd2')
            #     paramsC.directories.Train.Model_Thalamus = paramsC.directories.Train.Model_Thalamus.replace('sd0','sd2')
            #     paramsC.directories.Train.Model_3T       = paramsC.directories.Train.Model_3T.replace('sd0','sd2')
            #     choosingModel.check_Run(paramsC, Data)        
            
            
            Data, params = datasets.loadDataset(params)                
            choosingModel.check_Run(params, Data)  
            
            K.clear_session()
            
            # except: print('failed')
 

    if   UserInfoB['Model_Method'] == 'HCascade':  HierarchicalStages(UserInfoB)
    elif UserInfoB['Model_Method'] == 'Cascade' :  CacadeStages(UserInfoB)
    elif UserInfoB['Model_Method'] == 'singleRun': Run_SingleNuclei(UserInfoB)

def preMode(UserInfo):
    UserInfoB = smallFuncs.terminalEntries(UserInfo)
    params = paramFunc.Run(UserInfoB)
    datasets.movingFromDatasetToExperiments(params)
    applyPreprocess.main(params, 'experiment')
    K = gpuSetting(params)
    return UserInfoB, K
UserInfoB, K = preMode(UserInfo.__dict__)
UserInfoB['simulation'].verbose = 2


# try: 
#     UserInfoB['simulation'].nucleus_Index = [1]
#     IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
#     print('slicingDim' , IV.slicingDim , 'Nuclei_Indexes' , IV.Nuclei_Indexes , 'GPU:  ', UserInfoB['simulation'].GPU_Index)
#     Run(UserInfoB, IV)
# except:
#     print('----')


# try: 
    # UserInfoB['simulation'].nucleus_Index = [2,4,5,6,7,8,9,10,11,12,13,14]
IV = InitValues( UserInfoB['simulation'].nucleus_Index , UserInfoB['simulation'].slicingDim)
print('slicingDim' , IV.slicingDim , 'Nuclei_Indexes' , IV.Nuclei_Indexes , 'GPU:  ', UserInfoB['simulation'].GPU_Index)
Run(UserInfoB, IV)
# except:
    # print('----')


# K.clear_session()
